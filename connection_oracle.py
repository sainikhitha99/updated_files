"""
Oracle database connection module.
Handles secure connection and operations for Oracle databases.
"""

import re
import os
import glob
import shutil
import logging
from typing import Optional, Dict, Any, List, Tuple
from database_connection import DatabaseConnection

logger = logging.getLogger(__name__)


def _parse_tns(tns: str) -> Dict[str, str]:
    """Extract host, port, service_name, and sid from a TNS descriptor string."""
    result: Dict[str, str] = {}
    host_m = re.search(r'HOST\s*=\s*([^\s()]+)', tns, re.IGNORECASE)
    if host_m:
        result['host'] = host_m.group(1).strip()
    port_m = re.search(r'PORT\s*=\s*(\d+)', tns, re.IGNORECASE)
    if port_m:
        result['port'] = port_m.group(1).strip()
    svc_m = re.search(r'SERVICE_NAME\s*=\s*([^\s()]+)', tns, re.IGNORECASE)
    if svc_m:
        result['service_name'] = svc_m.group(1).strip()
    sid_m = re.search(r'\bSID\s*=\s*([^\s()]+)', tns, re.IGNORECASE)
    if sid_m:
        result['sid'] = sid_m.group(1).strip()
    return result


THICK_MODE_ENABLED = False


def _oracle_client_dirs() -> List[str]:
    """Return candidate Oracle client library directories for thick mode."""
    candidates: List[str] = []

    env_lib = os.getenv("ORACLE_CLIENT_LIB_DIR", "").strip()
    if env_lib:
        candidates.append(env_lib)

    oracle_home = os.getenv("ORACLE_HOME", "").strip()
    if oracle_home:
        candidates.append(os.path.join(oracle_home, "bin"))

    sqlplus_path = shutil.which("sqlplus")
    if sqlplus_path:
        candidates.append(os.path.dirname(sqlplus_path))

    if os.name == "nt":
        candidates.extend(sorted(glob.glob(r"C:\\oracle\\instantclient_*"), reverse=True))
        candidates.extend(sorted(glob.glob(r"C:\\app\\*\\product\\*\\client_*\\bin"), reverse=True))

    # Deduplicate while preserving order.
    unique: List[str] = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm not in unique and os.path.isdir(norm):
            unique.append(norm)
    return unique


def _ensure_thick_mode() -> bool:
    """Try to enable python-oracledb thick mode and return True on success."""
    global THICK_MODE_ENABLED
    if THICK_MODE_ENABLED:
        return True

    if not ORACLE_AVAILABLE:
        return False

    # Try explicit client directories first because VS Code process PATH may differ
    # from an interactive shell where sqlplus works.
    for lib_dir in _oracle_client_dirs():
        try:
            oracledb.init_oracle_client(lib_dir=lib_dir)
            THICK_MODE_ENABLED = True
            logger.info(f"oracledb thick mode initialized using Oracle client at: {lib_dir}")
            return True
        except Exception as e:
            logger.debug(f"Thick mode init failed for '{lib_dir}': {e}")

    # Final fallback: default loader lookup.
    try:
        oracledb.init_oracle_client()
        THICK_MODE_ENABLED = True
        logger.info("oracledb thick mode initialized using default Oracle client lookup")
        return True
    except Exception as e:
        logger.debug(f"oracledb thick mode not available, using thin mode: {e}")
        return False


try:
    import oracledb  # type: ignore[import-not-found]
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    logger.warning("oracledb not installed. Oracle support unavailable. Install with: pip install oracledb")

if ORACLE_AVAILABLE:
    _ensure_thick_mode()


class OracleConnection(DatabaseConnection):
    """Manages Oracle database connections."""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str,
                 service_name: Optional[str] = None,
                 use_sid: bool = False, pdb_name: Optional[str] = None,
                 tns_dsn: Optional[str] = None):
        """
        Initialize Oracle connection.

        Args:
            host: Database hostname
            port: Database port (default 1521)
            database: Database SID or service name
            user: Database user
            password: Database password
            service_name: Optional service name (if different from database)
            use_sid: Use SID-based connection (full TNS descriptor) instead of SERVICE_NAME Easy Connect
            pdb_name: CDB/PDB support — name of the Pluggable Database to switch into after
                      connecting to the CDB root (e.g. "ORCLPDB1", "FINPDB").
                      Use this when:
                        - Connecting via SID (which lands at CDB$ROOT)
                        - You want to analyze a specific PDB, not the whole CDB
                      Requires the user to have the CDB_DBA role or
                      SET CONTAINER privilege on the target PDB.
                      If pdb_name is NOT supplied but the database IS a CDB, metrics are
                      collected from CDB$ROOT scope (all containers).
            tns_dsn: Full TNS descriptor string to use as-is for the DSN.
                     Example:
                       "(DESCRIPTION=(ADDRESS=(PROTOCOL=tcp)(HOST=u6scm1c1.ffdc.sbc.com)(PORT=12099))"
                       "(CONNECT_DATA=(SERVICE_NAME=SCMUAT3)))"
                     When supplied, overrides host/port/database DSN construction entirely.
                     host/port/database are still used as fallback labels in log messages.
                     Use this for complex TNS descriptors (RAC, failover, TCPS, LDAP etc.).
        """
        if not ORACLE_AVAILABLE:
            raise ImportError("oracledb not installed. Install with: pip install oracledb")

        # If a raw TNS descriptor is provided, parse host/port/service from it for labelling;
        # the actual connection will always use tns_dsn verbatim.
        if tns_dsn:
            parsed = _parse_tns(tns_dsn)
            host = parsed.get('host') or host or 'unknown'
            port = int(parsed.get('port') or port or 1521)
            database = parsed.get('service_name') or parsed.get('sid') or database or 'unknown'

        super().__init__(host, port or 1521, database, user, password)
        self.service_name = service_name or database
        self.use_sid = use_sid
        self.pdb_name = pdb_name       # PDB to switch into after connecting to CDB root
        self.tns_dsn = tns_dsn         # Raw TNS descriptor; takes precedence over all DSN building
        self.connection = None
        self.last_error: Optional[str] = None
        self.is_cdb: Optional[bool] = None        # True once connected and CDB detected
        self.current_container: Optional[str] = None  # Name of the active container

    def connect(self) -> bool:
        """
        Establish connection to Oracle database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Priority: raw TNS descriptor > SID descriptor > Easy Connect (service name)
            if self.tns_dsn:
                dsn = self.tns_dsn.strip()
                logger.info(
                    f"Oracle TNS DSN supplied directly. "
                    f"Target: {self.host}:{self.port}/{self.service_name}"
                )
            elif self.use_sid:
                dsn = (f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={self.host})(PORT={self.port}))"
                       f"(CONNECT_DATA=(SID={self.service_name})))")
                logger.info(f"Oracle SID DSN (sqlplus equivalent: sqlplus {self.user}/***@{self.host}:{self.port}/{self.service_name})")
            else:
                dsn = f"{self.host}:{self.port}/{self.service_name}"
                logger.info(f"Oracle DSN: {dsn}  (sqlplus equivalent: sqlplus {self.user}/***@{dsn})")
            connect_kwargs = {
                'user': self.user,
                'password': self.password,
                'dsn': dsn,
            }
            try:
                self.connection = oracledb.connect(**connect_kwargs)
            except Exception as e:
                err_msg = str(e)
                if "DPY-3015" in err_msg:
                    # If we hit DPY-3015, ensure thick mode and retry once.
                    if _ensure_thick_mode():
                        self.connection = oracledb.connect(**connect_kwargs)
                    else:
                        self.last_error = (
                            "DPY-3015 in thin mode. Install Oracle Instant Client and set "
                            "ORACLE_CLIENT_LIB_DIR, then restart the service. Original error: "
                            f"{err_msg}"
                        )
                        logger.error(self.last_error)
                        self.is_connected = False
                        return False
                else:
                    raise
            logger.info(f"Successfully connected to Oracle at {self.host}:{self.port}/{self.service_name}")
            self.is_connected = True

            # Set call timeout (30 seconds per round-trip) to prevent queries from hanging
            try:
                self.connection.call_timeout = 30000  # milliseconds
            except Exception:
                pass  # call_timeout may not be available in all oracledb versions

            # Test the connection
            is_alive, msg = self.test_connection()
            if not is_alive:
                logger.warning(f"Connection test failed: {msg}")
                self.close()
                return False

            # Detect whether this instance is a CDB
            self._detect_cdb()

            # If caller requested a specific PDB, switch to it now
            if self.pdb_name:
                if not self.is_cdb:
                    logger.warning(
                        f"pdb_name='{self.pdb_name}' was specified but this instance is not a CDB. "
                        "Ignoring container switch — proceeding as non-CDB connection."
                    )
                else:
                    ok, err = self._switch_to_pdb(self.pdb_name)
                    if not ok:
                        logger.error(f"Failed to switch to PDB '{self.pdb_name}': {err}")
                        self.close()
                        self.last_error = err
                        return False

            return True
        except Exception as e:
            logger.error(f"Oracle connection failed: {e}")
            self.last_error = str(e)
            self.is_connected = False
            return False

    # ------------------------------------------------------------------
    # CDB / PDB helpers
    # ------------------------------------------------------------------

    def _detect_cdb(self) -> None:
        """Detect whether the connected Oracle instance is a CDB and set self.is_cdb / self.current_container."""
        try:
            result = self.execute_query_dict(
                "SELECT d.cdb, "
                "SYS_CONTEXT('USERENV','CON_NAME') AS con_name "
                "FROM v$database d"
            )
            if result:
                row = result[0]
                cdb_val = str(row.get('CDB') or row.get('cdb') or 'NO').upper()
                self.is_cdb = (cdb_val == 'YES')
                self.current_container = str(
                    row.get('CON_NAME') or row.get('con_name') or 'CDB$ROOT'
                )
                logger.info(
                    f"CDB detection: is_cdb={self.is_cdb}, "
                    f"current_container={self.current_container}"
                )
        except Exception as e:
            logger.warning(f"CDB detection failed (non-critical): {e}")
            self.is_cdb = False
            self.current_container = None

    def _switch_to_pdb(self, pdb_name: str) -> Tuple[bool, str]:
        """
        Switch the session context to a specific PDB using ALTER SESSION SET CONTAINER.

        Requires: SYSDBA privilege or SET CONTAINER privilege on the target PDB.
        This is the correct approach when you connected via the CDB SID and want
        to run queries scoped to a single PDB.

        Args:
            pdb_name: Name of the PDB (e.g. 'ORCLPDB1')

        Returns:
            (True, 'Switched to <pdb_name>') on success
            (False, error_message) on failure
        """
        # Sanitize PDB name: only allow alphanumeric, underscore, dollar sign
        import re as _re
        if not _re.match(r'^[A-Za-z][A-Za-z0-9_$#]{0,127}$', pdb_name):
            return False, f"Invalid PDB name: '{pdb_name}' — must be a valid Oracle identifier"
        try:
            logger.info(f"Attempting to switch to PDB: {pdb_name}")
            cursor = self.connection.cursor()
            cursor.execute(f"ALTER SESSION SET CONTAINER = {pdb_name}")
            cursor.close()
            
            # Verify the switch worked by checking the current container
            verify_cursor = self.connection.cursor()
            verify_cursor.execute("SELECT SYS_CONTEXT('USERENV','CON_NAME') AS con_name, SYS_CONTEXT('USERENV','CON_ID') AS con_id FROM dual")
            verify_row = verify_cursor.fetchone()
            verify_cursor.close()
            
            if verify_row:
                actual_con_name = verify_row[0]
                actual_con_id = verify_row[1]
                logger.info(f"✓ Successfully switched to PDB: {actual_con_name} (CON_ID={actual_con_id})")
                self.current_container = str(actual_con_name).upper()
                return True, f"Switched to {actual_con_name}"
            else:
                self.current_container = pdb_name.upper()
                logger.info(f"Switched to {pdb_name} (verification query returned no rows)")
                return True, f"Switched to {pdb_name}"
        except Exception as e:
            err_str = str(e)
            logger.error(f"✗ Failed to switch to PDB '{pdb_name}': {err_str}")
            return False, err_str

    def list_pluggable_databases(self) -> List[Dict[str, Any]]:
        """
        List all PDBs visible from the current connection (requires CDB).

        Returns a list of dicts with keys: PDB_ID, PDB_NAME, STATUS, OPEN_MODE.
        Returns empty list if not a CDB or insufficient privileges.
        """
        if not self.is_cdb:
            return []
        try:
            return self.execute_query_dict(
                "SELECT pdb_id, pdb_name, status, open_mode "
                "FROM cdb_pdbs ORDER BY pdb_name"
            )
        except Exception as e:
            logger.warning(f"Could not query cdb_pdbs: {e}")
            # Fallback: v$pdbs (requires SYSDBA)
            try:
                return self.execute_query_dict(
                    "SELECT con_id AS pdb_id, name AS pdb_name, open_mode "
                    "FROM v$pdbs ORDER BY name"
                )
            except Exception as e2:
                logger.error(f"Could not list PDBs: {e2}")
                return []

    def get_cdb_info(self) -> Dict[str, Any]:
        """
        Return a summary of CDB/PDB architecture info for the current connection.

        Useful for diagnostics and the health-check report.
        """
        if not self.is_cdb:
            return {'multitenant': False, 'current_container': self.current_container}
        pdbs = self.list_pluggable_databases()
        return {
            'multitenant': True,
            'current_container': self.current_container,
            'pdb_count': len(pdbs),
            'pdbs': [
                {
                    'name': str(p.get('PDB_NAME') or p.get('pdb_name') or ''),
                    'status': str(p.get('STATUS') or p.get('status') or ''),
                    'open_mode': str(p.get('OPEN_MODE') or p.get('open_mode') or ''),
                }
                for p in pdbs
            ],
        }
    
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """
        Execute a SELECT query and return results as tuples.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of tuples with results
        """
        if not self.is_connected or not self.connection:
            logger.error("Not connected to database")
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or {})
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    # execute_ddl intentionally removed — this tool is strictly read-only.
    # The only session-level command allowed is ALTER SESSION SET CONTAINER (handled in _switch_to_pdb).

    def execute_query_dict(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as dictionaries.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            List of dictionaries with results
        """
        if not self.is_connected or not self.connection:
            logger.error("Not connected to database")
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or {})
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch results and convert to dicts
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    def execute_batch_queries_dict(
        self, queries: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute multiple SQL queries in a single persistent cursor session.

        Unlike the SQLcl/MCP approach, this uses the native oracledb cursor
        directly — no JVM startup, no spool files, no script buffer limits.
        Each query is executed sequentially via the same connection; a failure
        in one query does not abort the others.

        Args:
            queries: Mapping of label -> SQL text.

        Returns:
            Mapping of label -> list of row dicts.
            A query that errors returns an empty list for that label.
        """
        results: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in queries}
        if not self.is_connected or not self.connection:
            logger.error("execute_batch_queries_dict: not connected")
            return results
        return self._run_batch_on(self.connection, queries)

    @staticmethod
    def _run_batch_on(
        connection, queries: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run a label->SQL batch on an oracledb connection.

        Each query runs sequentially on one cursor; a failure in one does not
        abort the others.
        """
        import time as _time

        results: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in queries}
        if connection is None:
            return results

        ok_count = 0
        err_labels: List[str] = []
        batch_start = _time.time()
        cursor = connection.cursor()
        try:
            for label, sql_text in queries.items():
                sql = (sql_text or "").strip().rstrip(";")
                if not sql:
                    continue
                try:
                    cursor.execute(sql)
                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        rows = []
                        for row in cursor.fetchall():
                            row_dict: Dict[str, Any] = dict(zip(columns, row))
                            # Add uppercase aliases for compatibility
                            for k, v in list(row_dict.items()):
                                up = k.upper()
                                if up not in row_dict:
                                    row_dict[up] = v
                            rows.append(row_dict)
                        results[label] = rows
                    ok_count += 1
                except Exception as qe:
                    err_labels.append(label)
                    logger.debug("Batch query '%s' failed: %s", label, qe)
        finally:
            cursor.close()

        elapsed_ms = int((_time.time() - batch_start) * 1000)
        if err_labels:
            logger.info(
                "Batch native: %d OK, %d errors (%s) in %d ms",
                ok_count, len(err_labels), ",".join(err_labels[:15]), elapsed_ms,
            )
        else:
            logger.info("Batch native: %d/%d OK in %d ms", ok_count, len(queries), elapsed_ms)
        return results
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test if connection is alive."""
        try:
            if not self.connection:
                return False, "Not connected"
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1 FROM dual")
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return True, "Connection successful"
            return False, "Connection test returned no results"
        except Exception as e:
            return False, f"Connection test failed: {e}"
    
    def get_database_type(self) -> str:
        """Return database type identifier."""
        return "oracle"
    
    def get_version(self) -> str:
        """Get Oracle database version."""
        # Try v$instance first — returns the compact version string e.g. "19.3.0.0.0"
        try:
            result = self.execute_query_dict("SELECT version FROM v$instance")
            if result:
                val = result[0].get('VERSION') or result[0].get('version')
                if val:
                    return str(val)
        except Exception:
            pass
        # Fallback: full banner from v$version
        try:
            results = self.execute_query_dict("SELECT banner FROM v$version WHERE ROWNUM = 1")
            if results:
                return str(results[0].get('BANNER') or results[0].get('banner') or 'Unknown')
        except Exception as e:
            logger.error(f"Failed to get version: {e}")
        return "Unknown"
    
    def get_database_size(self) -> Tuple[str, int]:
        """Get Oracle database size."""
        try:
            query = """
            SELECT 
                ROUND(SUM(bytes)/1024/1024/1024, 2) as size_gb,
                SUM(bytes) as size_bytes
            FROM dba_data_files
            """
            results = self.execute_query_dict(query)
            if results:
                row = results[0]
                # Oracle returns column names in UPPERCASE
                size_bytes = int(row.get('SIZE_BYTES') or row.get('size_bytes') or 0)
                size_gb = float(row.get('SIZE_GB') or row.get('size_gb') or 0)
                return f"{size_gb} GB", size_bytes
        except Exception as e:
            logger.warning(f"Could not get database size: {e}")
        
        return "Unknown", 0
    
    def close(self) -> bool:
        """Close database connection."""
        try:
            if self.connection:
                self.connection.close()
                self.is_connected = False
                logger.info("Oracle connection closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close Oracle connection: {e}")
            return False


__all__ = ['OracleConnection']
