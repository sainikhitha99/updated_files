"""
PostgreSQL performance metrics collection module.
Gathers comprehensive database performance statistics.
"""

import logging
import re
import time
from threading import Lock
from typing import Dict, List, Any, Optional, Tuple
from database_connection import DatabaseConnection
from utils import _scalar, _safe_int, _safe_float

logger = logging.getLogger(__name__)


# ── Module-level helpers imported from utils.py ──────────────────────────────
# _scalar, _safe_int, _safe_float are imported above.


# Short-TTL cache for Oracle MCP metrics to avoid re-spawning 18+ JVM processes
# on every consecutive health-check / ask-question within the same minute.
_metrics_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_METRICS_CACHE_TTL = 60  # seconds
_METRICS_CACHE_MAX = 20  # max connections to cache (evict oldest when exceeded)
_metrics_cache_lock = Lock()  # thread-safety for concurrent cache access


def _metrics_cache_put(key: str, value: Dict[str, Any]) -> None:
    """Store a metrics snapshot in the cache, evicting stale/oldest entries if full."""
    now = time.monotonic()
    with _metrics_cache_lock:
        # Evict expired entries first
        expired = [k for k, (ts, _) in _metrics_cache.items() if now - ts >= _METRICS_CACHE_TTL]
        for k in expired:
            del _metrics_cache[k]
        # If still at capacity, evict the oldest entry
        if len(_metrics_cache) >= _METRICS_CACHE_MAX:
            oldest_key = min(_metrics_cache, key=lambda k: _metrics_cache[k][0])
            del _metrics_cache[oldest_key]
        _metrics_cache[key] = (now, value)


def _oracle_top_sql_for_batch(limit: int = 20) -> str:
    """Return the unified top-SQL query string for use inside batch dicts."""
    from api import _ORACLE_TOP_SQL_QUERY
    return _ORACLE_TOP_SQL_QUERY.format(limit=limit)


def _metrics_cache_key(conn: DatabaseConnection) -> str:
    """Build a cache key from connection identity attributes."""
    return f"{conn.host}:{conn.port}/{conn.database}/{conn.user}"


class MetricsCollector:
    """Collects PostgreSQL performance metrics."""
    
    def __init__(self, conn: DatabaseConnection):
        """Initialize metrics collector with database connection."""
        self.conn = conn
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all available performance metrics.
        
        Returns:
            Dictionary with all metrics organized by category
        """
        try:
            metrics = {}
            db_type = self.conn.get_database_type().lower()

            # Short-TTL (60s) cache of the full metrics snapshot — applies to ALL
            # connection types. Collecting every category is expensive (Oracle MCP
            # spawns JVMs; PostgreSQL/native-Oracle issue 15-20 sequential round
            # trips), and the Ask-Questions tab, health endpoint, and recommendations
            # all request it back-to-back. Callers treat the result as read-only.
            try:
                cache_key = _metrics_cache_key(self.conn)
            except Exception:
                cache_key = None
            if cache_key is not None:
                now = time.monotonic()
                with _metrics_cache_lock:
                    cached = _metrics_cache.get(cache_key)
                if cached is not None and 0 <= (now - cached[0]) < _METRICS_CACHE_TTL:
                    logger.info("Returning cached collect_all_metrics (TTL %ds)", _METRICS_CACHE_TTL)
                    return cached[1]

            if db_type == 'oracle':
                _t_total = time.monotonic()
                logger.info("Collecting Oracle metrics...")

                # ── Try batched single-JVM collection (MCP connections) ──
                if hasattr(self.conn, 'execute_batch_queries_dict'):
                    try:
                        metrics = self._collect_all_oracle_metrics_batched()
                        _total_ms = (time.monotonic() - _t_total) * 1000
                        logger.info("Oracle collect_all_metrics (batched) total: %.0f ms", _total_ms)
                        if cache_key is not None:
                            _metrics_cache_put(cache_key, metrics)
                        return metrics
                    except Exception as ex:
                        logger.warning("Batched Oracle collection failed, falling back to sequential: %s", ex)

                # ── Fallback: sequential per-category (1 JVM per query) ──
                _categories = [
                    ('general',      self._collect_oracle_general_metrics),
                    ('connections',  self._collect_oracle_connection_metrics),
                    ('cache',        self._collect_oracle_cache_metrics),
                    ('queries',      self._collect_oracle_query_metrics),
                    ('indexes',      self._collect_oracle_index_metrics),
                    ('tables',       self._collect_oracle_table_metrics),
                    ('locks',        self._collect_oracle_lock_metrics),
                    ('replication',  self._collect_oracle_replication_metrics),
                ]
                for _cat_name, _cat_fn in _categories:
                    _t0 = time.monotonic()
                    metrics[_cat_name] = _cat_fn()
                    _elapsed_ms = (time.monotonic() - _t0) * 1000
                    logger.info("Oracle metric [%s] collected in %.0f ms", _cat_name, _elapsed_ms)
                _total_ms = (time.monotonic() - _t_total) * 1000
                logger.info("Oracle collect_all_metrics total: %.0f ms", _total_ms)
                if cache_key is not None:
                    _metrics_cache_put(cache_key, metrics)
                return metrics

            # Default to PostgreSQL collector
            logger.info("Collecting PostgreSQL metrics...")
            logger.info("Collecting general metrics...")
            metrics['general'] = self._collect_general_metrics()
            logger.info("Collecting connection metrics...")
            metrics['connections'] = self._collect_connection_metrics()
            logger.info("Collecting cache metrics...")
            metrics['cache'] = self._collect_cache_metrics()
            logger.info("Collecting query metrics...")
            metrics['queries'] = self._collect_query_metrics()
            logger.info("Collecting index metrics...")
            metrics['indexes'] = self._collect_index_metrics()
            logger.info("Collecting table metrics...")
            metrics['tables'] = self._collect_table_metrics()
            logger.info("Collecting lock metrics...")
            metrics['locks'] = self._collect_lock_metrics()
            logger.info("Collecting replication metrics...")
            metrics['replication'] = self._collect_replication_metrics()
            if cache_key is not None:
                _metrics_cache_put(cache_key, metrics)
            return metrics

        except Exception as e:
            logger.error(f"Error during metrics collection: {e}", exc_info=True)
            raise
    
    def _collect_general_metrics(self) -> Dict[str, Any]:
        """Collect general database metrics."""
        # Get database version
        version_query = "SELECT version() as version;"
        version_result = self.conn.execute_query_dict(version_query)
        
        # Get database size
        size_query = """
        SELECT 
            pg_database.datname as database_name,
            pg_size_pretty(pg_database_size(pg_database.datname)) as database_size,
            pg_database_size(pg_database.datname)::bigint as size_bytes
        FROM pg_database 
        WHERE datname = current_database();
        """
        size_result = self.conn.execute_query_dict(size_query)
        
        return {
            'version': version_result[0]['version'] if version_result else 'Unknown',
            'database': size_result[0] if size_result else {}
        }
    
    def _collect_oracle_general_metrics(self) -> Dict[str, Any]:
        """Collect general Oracle metrics."""
        version = self.conn.get_version()
        size_human, size_bytes = self.conn.get_database_size()

        # Uptime from v$instance
        uptime = 'Unknown'
        try:
            result = self.conn.execute_query_dict(
                "SELECT (sysdate - startup_time)*24*60*60 AS uptime_seconds "
                "FROM v$instance"
            )
            if result and 'UPTIME_SECONDS' in result[0]:
                uptime_seconds = float(result[0]['UPTIME_SECONDS'])
                uptime = f"{uptime_seconds:.0f} seconds"
        except Exception:
            uptime = 'Unknown'

        # CDB/PDB architecture info (non-critical — graceful fallback)
        cdb_info: Dict[str, Any] = {}
        if hasattr(self.conn, 'get_cdb_info'):
            try:
                cdb_info = self.conn.get_cdb_info()
            except Exception:
                pass

        result: Dict[str, Any] = {
            'version': version,
            'database': {'size_human': size_human, 'size_bytes': size_bytes},
            'uptime': uptime,
        }
        if cdb_info:
            result['multitenant'] = cdb_info
        return result

    def _collect_oracle_connection_metrics(self) -> Dict[str, Any]:
        """Collect Oracle connection metrics."""
        connections = []
        active_connections = 0
        max_connections = 0

        try:
            connections = self.conn.execute_query_dict(
                "SELECT status, COUNT(*) as connection_count FROM v$session GROUP BY status"
            )
            active_row = self.conn.execute_query_dict(
                "SELECT COUNT(*) AS active_connections FROM v$session WHERE status='ACTIVE'"
            )
            active_connections = int(active_row[0]['ACTIVE_CONNECTIONS']) if active_row else 0
        except Exception:
            pass

        try:
            max_row = self.conn.execute_query_dict(
                "SELECT value FROM v$parameter WHERE name='sessions'"
            )
            max_connections = int(max_row[0]['VALUE']) if max_row else 0
        except Exception:
            max_connections = 0

        connection_usage = round((active_connections / max_connections * 100) if max_connections > 0 else 0, 2)

        return {
            'connections': connections,
            'max_connections': max_connections,
            'active_connections': active_connections,
            'connection_usage_percent': connection_usage
        }

    def _collect_oracle_cache_metrics(self) -> Dict[str, Any]:
        """Collect Oracle cache-related metrics."""
        overall_hit_ratio = 0
        try:
            result = self.conn.execute_query_dict(
                "SELECT ROUND(1 - (SUM(CASE name WHEN 'physical reads' THEN value ELSE 0 END) / "
                "NULLIF(SUM(CASE name WHEN 'db block gets' THEN value "
                "WHEN 'consistent gets' THEN value ELSE 0 END), 0)), 4) * 100 AS hit_ratio "
                "FROM v$sysstat "
                "WHERE name IN ('db block gets', 'consistent gets', 'physical reads')"
            )
            if result:
                val = result[0].get('HIT_RATIO') or result[0].get('hit_ratio')
                overall_hit_ratio = float(val or 0)
        except Exception:
            overall_hit_ratio = 0

        return {
            'table_cache_stats': [],
            'overall_hit_ratio': overall_hit_ratio
        }

    def _collect_oracle_query_metrics(self) -> Dict[str, Any]:
        """Collect Oracle query metrics."""
        slow_queries = []
        total_queries = 0
        try:
            from api import _ORACLE_TOP_SQL_QUERY, _filter_oracle_sys_sql
            slow_queries = self.conn.execute_query_dict(
                _ORACLE_TOP_SQL_QUERY.format(limit=40)
            )
            slow_queries = _filter_oracle_sys_sql(slow_queries)[:20]
            total_row = self.conn.execute_query_dict("SELECT COUNT(*) AS total_queries FROM v$sql")
            total_queries = int(total_row[0]['TOTAL_QUERIES']) if total_row else 0
        except Exception:
            pass

        return {
            'slow_queries': slow_queries,
            'total_queries': total_queries
        }
    
    def collect_top_sql_with_plans(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Collect top N SQL queries with their execution plans and optimization hints.
        
        Args:
            top_n: Number of top queries to analyze (default: 5)
            
        Returns:
            List of dictionaries with query details, execution plan, and hints
        """
        try:
            db_type = self.conn.get_database_type().lower()
            if db_type == 'oracle':
                return self._get_oracle_sql_plans_detailed(top_n)
            else:
                return self._get_postgresql_sql_plans_detailed(top_n)
        except Exception as e:
            logger.error(f"Error collecting SQL plans: {e}")
            return []
    
    def _get_oracle_sql_plans_detailed(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Fetch Oracle execution plans for top SQL queries.

        Optimized: batches all plan/alternate-plan queries into a single JVM
        session via ``execute_batch_queries_dict`` instead of spawning 6-15
        individual JVMs (one per DBMS_XPLAN call + one per alternate-plan query).
        """
        plans: List[Dict[str, Any]] = []

        try:
            # ── Step 1: get top SQL (unified weighted scoring) ──
            from api import _ORACLE_TOP_SQL_QUERY, _filter_oracle_sys_sql
            # Fetch extra rows (4x buffer) to compensate for Python post-filtering
            fetch_limit = max(top_n * 4, 20)
            top_sql = self.conn.execute_query_dict(
                _ORACLE_TOP_SQL_QUERY.format(limit=fetch_limit)
            )
            # Python post-filter: remove any SYS/internal SQL that bypassed Oracle WHERE
            top_sql = _filter_oracle_sys_sql(top_sql)[:top_n]
            if not top_sql:
                return plans

            # ── Step 2: build batch queries for plans + alternates ──
            batch_queries: Dict[str, str] = {}
            sql_entries: List[Dict[str, Any]] = []  # parallel list of entry dicts

            for idx, row in enumerate(top_sql, 1):
                sql_id = str(row.get('SQL_ID') or row.get('sql_id') or '').strip()
                if not sql_id:
                    continue

                sql_text = str(row.get('SQL_TEXT') or row.get('sql_text') or '')[:800]
                elapsed_ms = _safe_float(row.get('ELAPSED_TIME') or row.get('elapsed_time') or 0) / 1e6
                executions = _safe_int(row.get('EXECUTIONS') or row.get('executions') or 1, 1)
                buffer_gets = _safe_int(row.get('BUFFER_GETS') or row.get('buffer_gets') or 0)
                avg_elapsed = _safe_float(row.get('AVG_ELAPSED_SEC') or 0)
                current_plan_hash_raw = row.get('PLAN_HASH_VALUE') or row.get('plan_hash_value')
                force_sig_raw = row.get('FORCE_MATCHING_SIGNATURE') or row.get('force_matching_signature')
                current_plan_hash = int(current_plan_hash_raw) if current_plan_hash_raw is not None else None
                force_sig = int(force_sig_raw) if force_sig_raw is not None else None

                entry: Dict[str, Any] = {
                    'rank': idx,
                    'sql_id': sql_id,
                    'query_text': sql_text,
                    'executions': executions,
                    'total_elapsed_ms': round(elapsed_ms, 2),
                    'avg_elapsed_ms': round(avg_elapsed * 1000, 2),
                    'buffer_gets': buffer_gets,
                    'current_plan_hash_value': current_plan_hash,
                    'execution_plan': '',
                    'execution_plan_rows': [],
                    'optimization_hints': [],
                    'suggested_sql': [],
                    'alternate_plans': [],
                    'plan_comparison': {},
                    'plan_ml_confidence_pct': 0.0,
                    'force_matching_signature': force_sig,
                    'avg_elapsed_sec': avg_elapsed,
                    'metrics': {
                        'rows_per_execution': 0,
                        'buffer_gets_per_execution': round(buffer_gets / executions, 2) if executions > 0 else 0,
                    }
                }
                sql_entries.append(entry)

                # ── Plan queries (3-layer fallback per SQL) ──
                # Layer 1: DBMS_XPLAN — runtime stats; NULL picks best available child cursor
                batch_queries[f"plan_{sql_id}"] = (
                    f"SELECT PLAN_TABLE_OUTPUT FROM TABLE("
                    f"DBMS_XPLAN.DISPLAY_CURSOR('{sql_id}', NULL, 'ALLSTATS LAST'))"
                )
                # Layer 2: v$sql_plan — structured, min child (survives when DISPLAY_CURSOR is empty)
                batch_queries[f"vplan_{sql_id}"] = (
                    f"SELECT id, parent_id, operation, options, object_name, depth, cost, cardinality, "
                    f"access_predicates, filter_predicates "
                    f"FROM v$sql_plan WHERE sql_id = '{sql_id}' "
                    f"AND child_number = (SELECT MIN(child_number) FROM v$sql_plan WHERE sql_id = '{sql_id}') "
                    f"ORDER BY id"
                )
                # Layer 3: awr_root_sql_plan — AWR historical (survives DB restart / SGA flush)
                batch_queries[f"awrplan_{sql_id}"] = (
                    f"SELECT id, parent_id, operation, options, object_name, depth, cost, cardinality, "
                    f"access_predicates, filter_predicates "
                    f"FROM awr_root_sql_plan "
                    f"WHERE sql_id = '{sql_id}' "
                    f"AND plan_hash_value = (SELECT MIN(plan_hash_value) FROM awr_root_sql_plan WHERE sql_id = '{sql_id}') "
                    f"ORDER BY id"
                )

                # Queue alternate-plan query (only if we have force_matching_signature)
                if force_sig:
                    batch_queries[f"alt_{sql_id}"] = (
                        "SELECT sql_id, plan_hash_value, executions, "
                        "       ROUND(elapsed_time / NULLIF(executions, 0) / 1e6, 4) AS avg_elapsed_sec, "
                        "       ROUND(cpu_time / NULLIF(executions, 0) / 1e6, 4) AS avg_cpu_sec, "
                        "       last_active_time "
                        "FROM ( "
                        "  SELECT sql_id, plan_hash_value, executions, elapsed_time, cpu_time, last_active_time "
                        "  FROM v$sql "
                        f"  WHERE force_matching_signature = {int(force_sig)} "
                        "    AND plan_hash_value IS NOT NULL "
                        "  ORDER BY elapsed_time / NULLIF(executions, 0) "
                        ") WHERE rownum <= 10"
                    )

            if not sql_entries:
                return plans

            # ── Step 3: execute all plan queries in a SINGLE JVM session ──
            _t0 = time.monotonic()
            if hasattr(self.conn, 'execute_batch_queries_dict') and batch_queries:
                batch_results = self.conn.execute_batch_queries_dict(batch_queries)
            else:
                batch_results = {}
            logger.info("Oracle SQL plans batched: %d queries in 1 JVM, %.0f ms",
                        len(batch_queries), (time.monotonic() - _t0) * 1000)

            # ── Step 4: assemble results ──
            for entry in sql_entries:
                sql_id = entry['sql_id']
                force_sig = entry.get('force_matching_signature')
                avg_elapsed = entry.get('avg_elapsed_sec', 0)
                current_plan_hash = entry.get('current_plan_hash_value')
                executions = entry.get('executions', 0)

                # ── Parse execution plan — 3-layer fallback ──────────────────────────
                # Layer 1: DBMS_XPLAN cursor text (has actual row counts / elapsed per step)
                cursor_rows = batch_results.get(f"plan_{sql_id}", [])
                vplan_rows = batch_results.get(f"vplan_{sql_id}", [])
                awr_rows = batch_results.get(f"awrplan_{sql_id}", [])
                plan_text = '\n'.join(
                    str(r.get('PLAN_TABLE_OUTPUT') or r.get('plan_table_output') or '')
                    for r in cursor_rows
                ).strip()
                plan_source = 'DISPLAY_CURSOR'

                if not plan_text:
                    # Layer 2: v$sql_plan — structured rows from shared pool
                    if vplan_rows:
                        plan_text = self._format_structured_plan_as_text(
                            vplan_rows, 'Shared Pool (v$sql_plan)'
                        )
                        plan_source = 'v$sql_plan'

                if not plan_text:
                    # Layer 3: awr_root_sql_plan — AWR historical baseline
                    if awr_rows:
                        plan_text = self._format_structured_plan_as_text(
                            awr_rows, 'AWR History (awr_root_sql_plan)'
                        )
                        plan_source = 'awr_root_sql_plan'

                entry['execution_plan'] = plan_text
                entry['plan_source'] = plan_source
                if vplan_rows:
                    entry['execution_plan_rows'] = vplan_rows
                elif awr_rows:
                    entry['execution_plan_rows'] = awr_rows
                elif cursor_rows:
                    entry['execution_plan_rows'] = self._parse_xplan_text_to_rows(plan_text)

                if not entry.get('current_plan_hash_value') and plan_text:
                    m = re.search(r'Plan\s+hash\s+value:\s*(\d+)', plan_text, flags=re.IGNORECASE)
                    if m:
                        entry['current_plan_hash_value'] = int(m.group(1))
                        current_plan_hash = entry['current_plan_hash_value']

                # ─────── FOOLPROOF analysis: structured + cardinality errors + bind skew ───────
                plan_analysis = {}
                cardinality_errors = []
                bind_skew = {'has_skew': False}

                if plan_text:
                    # Old regex-based hints (still used for UI display)
                    hints, suggestions = self._analyze_oracle_execution_plan(plan_text, entry['query_text'])
                    entry['optimization_hints'] = hints
                    entry['suggested_sql'] = suggestions
                    
                    # NEW: Structured foolproof analysis
                    vplan_rows = batch_results.get(f"vplan_{sql_id}", [])
                    if vplan_rows:
                        plan_analysis = self._analyze_plan_structure_foolproof(vplan_rows)
                        cardinality_errors = self._detect_cardinality_errors(plan_text, vplan_rows)
                        bind_skew = self._detect_bind_variable_skew(sql_id)
                        
                        # Generate SPECIFIC recommendations (not generic)
                        specific_recs = self._generate_specific_recommendations(
                            plan_analysis,
                            cardinality_errors,
                            bind_skew,
                            entry.get('execution_stats', [])
                        )
                        entry['specific_recommendations'] = specific_recs
                        entry['plan_analysis'] = plan_analysis
                        entry['cardinality_errors'] = cardinality_errors
                        entry['bind_variable_skew'] = bind_skew
                else:
                    entry['execution_plan'] = (
                        f"Plan unavailable for SQL_ID={sql_id}. "
                        f"Run: SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR('{sql_id}', NULL, 'ALLSTATS LAST'));"
                    )
                    entry['plan_source'] = 'unavailable'
                    entry['optimization_hints'] = [
                        'Execution plan not found in cursor cache, shared pool, or AWR history'
                    ]

                # Parse alternate plans from batch
                alt_rows = batch_results.get(f"alt_{sql_id}", [])
                alt_plans: List[Dict[str, Any]] = []
                dedup: Dict[int, Dict[str, Any]] = {}
                for r in alt_rows:
                    phv_raw = r.get('PLAN_HASH_VALUE') or r.get('plan_hash_value')
                    try:
                        phv = int(phv_raw)
                    except Exception:
                        continue
                    if phv not in dedup:
                        dedup[phv] = {
                            'sql_id': str(r.get('SQL_ID') or r.get('sql_id') or sql_id),
                            'plan_hash_value': phv,
                            'executions': _safe_int(r.get('EXECUTIONS') or r.get('executions') or 0),
                            'avg_elapsed_sec': _safe_float(r.get('AVG_ELAPSED_SEC') or r.get('avg_elapsed_sec') or 0),
                            'avg_cpu_sec': _safe_float(r.get('AVG_CPU_SEC') or r.get('avg_cpu_sec') or 0),
                            'last_active_time': str(r.get('LAST_ACTIVE_TIME') or r.get('last_active_time') or ''),
                            'is_current_plan': bool(current_plan_hash is not None and phv == current_plan_hash),
                        }
                alt_plans = list(dedup.values())
                entry['alternate_plans'] = alt_plans

                # Compare plans
                plan_cmp = self._compare_oracle_plans(
                    current_avg_sec=avg_elapsed,
                    current_plan_hash=current_plan_hash,
                    current_executions=executions,
                    alternate_plans=alt_plans,
                )
                entry['plan_comparison'] = plan_cmp
                entry['plan_ml_confidence_pct'] = _safe_float(plan_cmp.get('confidence_pct') or 0.0)

                if plan_cmp.get('better_plan_found'):
                    imp = plan_cmp.get('estimated_improvement_pct', 0)
                    best_hash = plan_cmp.get('best_plan_hash_value')
                    entry['optimization_hints'].append(
                        f"Alternate plan detected (hash {best_hash}) with estimated {imp}% lower average elapsed time."
                    )
                    entry['suggested_sql'].append(
                        "-- Compare alternate plans and baseline by plan hash\n"
                        f"SELECT sql_id, plan_hash_value, executions, elapsed_time, cpu_time "
                        f"FROM v$sql WHERE force_matching_signature = {int(force_sig) if force_sig else 0} "
                        "ORDER BY elapsed_time/NULLIF(executions,0);"
                    )

                # Remove internal-only keys before returning
                entry.pop('force_matching_signature', None)
                entry.pop('avg_elapsed_sec', None)

                plans.append(entry)

        except Exception as e:
            logger.error(f"Error fetching Oracle SQL plans: {e}")

        return plans
    
    def _get_postgresql_sql_plans_detailed(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Fetch PostgreSQL execution plans for top SQL queries."""
        plans: List[Dict[str, Any]] = []
        
        try:
            # Get top SQL by total execution time
            top_sql = self.conn.execute_query_dict(
                f"SELECT query, calls, total_exec_time, mean_exec_time, max_exec_time, rows, stddev_exec_time "
                f"FROM pg_stat_statements "
                f"WHERE query NOT LIKE '%pg_stat_statements%' AND query NOT LIKE '%information_schema%' "
                f"ORDER BY total_exec_time DESC "
                f"LIMIT {top_n}"
            )
            
            for idx, row in enumerate(top_sql, 1):
                sql_text = str(row.get('query') or '')
                if len(sql_text) < 10:
                    continue
                
                calls = int(row.get('calls') or 1)
                total_ms = float(row.get('total_exec_time') or 0)
                mean_ms = float(row.get('mean_exec_time') or 0)
                max_ms = float(row.get('max_exec_time') or 0)
                rows_affected = int(row.get('rows') or 0)
                
                entry: Dict[str, Any] = {
                    'rank': idx,
                    'query_text': sql_text[:800],
                    'calls': calls,
                    'total_exec_ms': round(total_ms, 2),
                    'avg_exec_ms': round(mean_ms, 2),
                    'max_exec_ms': round(max_ms, 2),
                    'rows_affected': rows_affected,
                    'execution_plan': '',
                    'optimization_hints': [],
                    'suggested_sql': [],
                    'metrics': {
                        'rows_per_call': round(rows_affected / calls, 2) if calls > 0 else 0,
                        'avg_rows_per_call': round(rows_affected / calls, 2) if calls > 0 else 0,
                    }
                }
                
                # Fetch EXPLAIN plan
                try:
                    plan_rows = self.conn.execute_query_dict(
                        f"EXPLAIN (VERBOSE, ANALYZE, BUFFERS, FORMAT TEXT) {sql_text}"
                    )
                    plan_text = '\n'.join(
                        str(r.get('QUERY PLAN') or r.get('query plan') or '') 
                        for r in plan_rows
                    )
                    entry['execution_plan'] = plan_text
                    
                    # Analyze the plan and generate hints
                    hints, suggestions = self._analyze_postgresql_execution_plan(plan_text, sql_text)
                    entry['optimization_hints'] = hints
                    entry['suggested_sql'] = suggestions
                except Exception as ex:
                    logger.warning(f"Could not fetch plan for query: {ex}")
                    entry['execution_plan'] = f"Plan unavailable: {str(ex)[:200]}"
                    entry['optimization_hints'] = ['Run EXPLAIN (ANALYZE, BUFFERS) in your query tool for the actual plan']
                
                plans.append(entry)
                
        except Exception as e:
            logger.error(f"Error fetching PostgreSQL SQL plans: {e}")
        
        return plans
    
    def _analyze_oracle_execution_plan(self, plan_text: str, sql_text: str) -> tuple[List[str], List[str]]:
        """Analyze Oracle execution plan and return optimization hints and suggested SQL."""
        hints: List[str] = []
        suggestions: List[str] = []
        plan_lower = plan_text.lower()
        sql_lower = sql_text.lower()
        sql_upper = sql_text.upper()
        import re

        def _sanitize_ident(name: str) -> str:
            clean = re.sub(r'[^A-Za-z0-9_]', '_', (name or '').strip())
            clean = re.sub(r'_+', '_', clean).strip('_')
            return clean[:20] if clean else 'col'

        def _extract_full_scan_tables(text: str) -> List[str]:
            tables: List[str] = []
            # DBMS_XPLAN commonly renders as: TABLE ACCESS FULL | EMP
            # and sometimes includes owner prefixes or quoted identifiers.
            patterns = [
                r'TABLE ACCESS FULL\s*\|\s*"?([A-Za-z0-9_\$#]+)"?(?:\."?([A-Za-z0-9_\$#]+)"?)?',
                r'TABLE ACCESS FULL\s+[A-Za-z ]*\|\s*"?([A-Za-z0-9_\$#]+)"?(?:\."?([A-Za-z0-9_\$#]+)"?)?',
                r'TABLE ACCESS FULL\s+"?([A-Za-z0-9_\$#]+)"?(?:\."?([A-Za-z0-9_\$#]+)"?)?',
            ]
            for pat in patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    g1 = m.group(1)
                    g2 = m.group(2)
                    table = g2 or g1
                    if table:
                        table_u = table.upper()
                        if table_u not in tables:
                            tables.append(table_u)
            return tables

        def _extract_alias_map(sql: str) -> Dict[str, str]:
            alias_map: Dict[str, str] = {}
            # Basic FROM/JOIN alias parsing for common SQL forms.
            for m in re.finditer(
                r'\b(?:from|join)\s+([a-zA-Z0-9_\$#\.\"]+)\s+(?:as\s+)?([a-zA-Z0-9_\$#\"]+)',
                sql,
                flags=re.IGNORECASE,
            ):
                tbl_raw = m.group(1).strip('"')
                alias_raw = m.group(2).strip('"')
                table_name = tbl_raw.split('.')[-1].strip('"').upper()
                alias = alias_raw.upper()
                if table_name and alias:
                    alias_map[alias] = table_name
            return alias_map

        def _extract_candidate_columns(sql: str, table: str, alias_map: Dict[str, str]) -> List[str]:
            table_u = table.upper()
            cols: List[str] = []

            # table.column patterns
            for m in re.finditer(r'\b([a-zA-Z0-9_\$#]+)\.([a-zA-Z0-9_\$#]+)\b', sql, flags=re.IGNORECASE):
                left = m.group(1).upper()
                col = m.group(2).upper()
                mapped = alias_map.get(left, left)
                if mapped == table_u and col not in cols:
                    cols.append(col)

            # Unqualified columns in simple single-table filters.
            if not cols:
                simple_where = re.search(r'\bwhere\b\s+(.*?)(?:\bgroup\b|\border\b|\bhaving\b|$)', sql, flags=re.IGNORECASE | re.DOTALL)
                if simple_where:
                    clause = simple_where.group(1)
                    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_\$#]*)\s*(=|>|<|>=|<=|like|in\b)', clause, flags=re.IGNORECASE):
                        col = m.group(1).upper()
                        if col not in cols and col not in {'AND', 'OR', 'NOT', 'IN', 'LIKE'}:
                            cols.append(col)

            return cols[:3]

        def _extract_predicate_profiles(text: str) -> Dict[str, Dict[str, List[str]]]:
            """Parse DBMS_XPLAN predicate section and classify columns by operator type.

            Returns a map keyed by operation id string, each containing:
              - equality_cols
              - range_cols
              - function_cols
              - all_cols
            """
            profiles: Dict[str, Dict[str, List[str]]] = {}
            section_match = re.search(
                r'Predicate Information.*?-+\s*(.*?)(?:\n\s*Note\b|\Z)',
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not section_match:
                return profiles

            section = section_match.group(1)
            entries = re.finditer(r'\n\s*(\d+)\s*-\s*filter\((.*?)\)(?=\n\s*\d+\s*-|\Z)', section, flags=re.IGNORECASE | re.DOTALL)
            for m in entries:
                op_id = m.group(1)
                expr = m.group(2)
                cols = [c.upper() for c in re.findall(r'"[A-Za-z0-9_\$#]+"\."([A-Za-z0-9_\$#]+)"', expr)]
                # Also capture standalone quoted columns like "YODA_ORDER_NUMBER".
                cols += [
                    c.upper() for c in re.findall(
                        r'(?<!\.)"([A-Za-z_][A-Za-z0-9_\$#]*)"(?!\s*\.)',
                        expr,
                    )
                ]
                unique_cols: List[str] = []
                for c in cols:
                    if c not in unique_cols:
                        unique_cols.append(c)

                eq_cols: List[str] = []
                range_cols: List[str] = []
                function_cols: List[str] = []

                for c in unique_cols:
                    if re.search(rf'"{re.escape(c)}"\s*=\s*', expr, flags=re.IGNORECASE):
                        eq_cols.append(c)
                    if re.search(rf'"{re.escape(c)}"\s*(>|<|>=|<=|\bbetween\b|\blike\b|\bin\b)', expr, flags=re.IGNORECASE):
                        range_cols.append(c)
                    if re.search(rf'INTERNAL_FUNCTION\([^\)]*"{re.escape(c)}"', expr, flags=re.IGNORECASE):
                        function_cols.append(c)

                profiles[op_id] = {
                    'equality_cols': eq_cols,
                    'range_cols': range_cols,
                    'function_cols': function_cols,
                    'all_cols': unique_cols,
                }

            return profiles

        def _choose_predicate_columns_for_full_scan(table: str, text: str, sql: str, alias_map: Dict[str, str]) -> Dict[str, List[str]]:
            """Choose best predicate columns for a full-scan table.

            Prefers operation-level predicates from plan text (DBMS_XPLAN), then SQL-text inference.
            """
            full_scan_ops = [
                m.group(1)
                for m in re.finditer(
                    rf'\|\s*(\d+)\s*\|\s*TABLE ACCESS FULL\s*\|\s*"?{re.escape(table)}"?\s*\|',
                    text,
                    flags=re.IGNORECASE,
                )
            ]

            profiles = _extract_predicate_profiles(text)
            eq_cols: List[str] = []
            range_cols: List[str] = []
            function_cols: List[str] = []
            all_cols: List[str] = []

            for op in full_scan_ops:
                p = profiles.get(op)
                if not p:
                    continue
                for c in p.get('equality_cols', []):
                    if c not in eq_cols:
                        eq_cols.append(c)
                for c in p.get('range_cols', []):
                    if c not in range_cols:
                        range_cols.append(c)
                for c in p.get('function_cols', []):
                    if c not in function_cols:
                        function_cols.append(c)
                for c in p.get('all_cols', []):
                    if c not in all_cols:
                        all_cols.append(c)

            if not all_cols and len(_extract_full_scan_tables(text)) == 1:
                # Robust fallback for complex DBMS_XPLAN formatting.
                # If there is a single full-scan table, infer predicate columns from plan text.
                inferred_cols = [
                    c.upper() for c in re.findall(
                        r'"[A-Za-z0-9_\$#]+"\."([A-Za-z0-9_\$#]+)"',
                        text,
                        flags=re.IGNORECASE,
                    )
                ]
                inferred_cols += [
                    c.upper() for c in re.findall(
                        r'(?<!\.)"([A-Za-z_][A-Za-z0-9_\$#]*)"(?!\s*\.)',
                        text,
                        flags=re.IGNORECASE,
                    )
                ]
                for c in inferred_cols:
                    if c not in all_cols:
                        all_cols.append(c)
                    if re.search(rf'"{re.escape(c)}"\s*=\s*', text, flags=re.IGNORECASE) and c not in eq_cols:
                        eq_cols.append(c)
                    if re.search(rf'"{re.escape(c)}"\s*(>|<|>=|<=|\bbetween\b|\blike\b|\bin\b)', text, flags=re.IGNORECASE) and c not in range_cols:
                        range_cols.append(c)
                    if re.search(rf'INTERNAL_FUNCTION\([^\)]*"{re.escape(c)}"', text, flags=re.IGNORECASE) and c not in function_cols:
                        function_cols.append(c)

            if not all_cols:
                sql_cols = _extract_candidate_columns(sql, table, alias_map)
                all_cols = sql_cols

            return {
                'equality_cols': eq_cols,
                'range_cols': range_cols,
                'function_cols': function_cols,
                'all_cols': all_cols,
            }
        
        # Full table scan detection
        full_scans = plan_text.count('TABLE ACCESS FULL')
        full_scan_tables = _extract_full_scan_tables(plan_text)
        if full_scans > 0:
            hints.append(f"⚠️  {full_scans} Full Table Scan(s) detected — review WHERE columns and add selective indexes")
            alias_map = _extract_alias_map(sql_text)
            if full_scan_tables:
                for tbl in full_scan_tables[:4]:
                    pred = _choose_predicate_columns_for_full_scan(tbl, plan_text, sql_text, alias_map)
                    eq_cols = pred.get('equality_cols', [])
                    range_cols = pred.get('range_cols', [])
                    function_cols = pred.get('function_cols', [])
                    candidate_cols = pred.get('all_cols', [])

                    if candidate_cols:
                        hints.append(f"Full Table Scan on `{tbl}` — predicate columns detected: {', '.join(candidate_cols)}")

                        # Build multiple index plans and rank them.
                        plan_options: List[Dict[str, str]] = []

                        if eq_cols or range_cols:
                            leading = eq_cols[:] if eq_cols else []
                            if range_cols:
                                for rc in range_cols:
                                    if rc not in leading:
                                        leading.append(rc)
                            if leading:
                                idx_name = f"idx_{_sanitize_ident(tbl.lower())}_{_sanitize_ident(leading[0].lower())}"
                                plan_options.append({
                                    'name': 'Plan A (Recommended)',
                                    'why': 'Best default for this plan: equality predicates first, then range predicates for selectivity.',
                                    'sql': f"CREATE INDEX {idx_name} ON {tbl}({', '.join(leading[:4])});",
                                })

                        if function_cols:
                            fc = function_cols[0]
                            idx_name = f"idx_{_sanitize_ident(tbl.lower())}_fb_{_sanitize_ident(fc.lower())}"
                            plan_options.append({
                                'name': 'Plan B',
                                'why': f"Use function-based indexing for transformed predicate column `{fc}` when SQL rewrite is not possible.",
                                'sql': (
                                    f"-- Replace UPPER() with the exact function used in predicate if different\n"
                                    f"CREATE INDEX {idx_name} ON {tbl}(UPPER({fc}));"
                                ),
                            })

                        if range_cols:
                            rc = range_cols[0]
                            idx_name = f"idx_{_sanitize_ident(tbl.lower())}_{_sanitize_ident(rc.lower())}_rng"
                            plan_options.append({
                                'name': 'Plan C',
                                'why': f"Use when range branch dominates and OR predicates can be rewritten/split. Targets `{rc}` scans.",
                                'sql': f"CREATE INDEX {idx_name} ON {tbl}({rc});",
                            })

                        if plan_options:
                            hints.append(
                                f"Optimization alternatives for `{tbl}`: "
                                f"{', '.join(p['name'] for p in plan_options)}. "
                                f"Best option: {plan_options[0]['name']}"
                            )
                            suggestions.append(f"-- Review WHERE/JOIN predicates for {tbl} and create a selective index")
                            for p in plan_options:
                                suggestions.append(f"-- {p['name']}: {p['why']}")
                                suggestions.append(p['sql'])
                    else:
                        hints.append(f"Full Table Scan on `{tbl}` — could not infer predicate columns from SQL text")
                        suggestions.append(f"-- Review WHERE/JOIN predicates for {tbl} and create a selective index")
            else:
                suggestions.append("-- Create index on filter columns\nCREATE INDEX idx_<table>_<column> ON <table>(<column>);")
        
        # Nested loops without index
        if 'NESTED LOOPS' in plan_text and 'INDEX' not in plan_text:
            hints.append("⚠️  Nested Loop Join without inner-side index — add indexes on JOIN columns")
        
        # High cost operations
        if 'SORT' in plan_text and ('ORDER BY' in sql_upper or 'GROUP BY' in sql_upper):
            hints.append("💡 Sort operation detected in plan — consider a covering index for ORDER BY/GROUP BY")
            suggestions.append("-- Create covering index\nCREATE INDEX idx_<table>_<columns> ON <table>(<order_by_cols>);")
        
        # Cartesian product warning
        if 'CARTESIAN' in plan_text or 'CROSS PRODUCT' in plan_text:
            hints.append("❌ CARTESIAN JOIN detected — verify JOIN condition exists, potential explosive row growth")
        
        # Cost analysis
        costs = [int(c) for c in re.findall(r'Cost[^\d]*(\d+)', plan_text) if c.isdigit()]
        if costs and max(costs) > 100000:
            hints.append(f"⚠️  Very high optimizer cost ({max(costs):,}) — review execution plan nodes for inefficiencies")
        
        # Parallel execution
        if 'PARALLEL' in plan_text:
            hints.append("ℹ️  Parallel execution plan — verify PARALLEL_MAX_SERVERS is optimally set")
        
        # Generate validation SQL
        suggestions.append("-- Verify with DBMS_XPLAN\nEXEC DBMS_XPLAN.DISPLAY_CURSOR;")
        if full_scan_tables:
            for tbl in full_scan_tables[:4]:
                suggestions.append(f"EXEC DBMS_STATS.GATHER_TABLE_STATS(ownname => USER, tabname => '{tbl}');")
        else:
            suggestions.append("-- Gather statistics if stale\nEXEC DBMS_STATS.GATHER_TABLE_STATS(USER, '<table>');")
        
        if not hints:
            hints.append("✅ Plan appears well-optimized — monitor execution time for any regressions")
        
        return hints, suggestions

    def _format_structured_plan_as_text(self, rows: List[Dict], source_label: str = '') -> str:
        """Render v$sql_plan / awr_root_sql_plan structured rows as DBMS_XPLAN-style text.

        Produces output whose patterns (TABLE ACCESS FULL, NESTED LOOPS, etc.) are
        recognised by ``_analyze_oracle_execution_plan``.
        """
        if not rows:
            return ''

        def _gv(r: Dict, *keys):
            for k in keys:
                v = r.get(k.upper()) or r.get(k.lower())
                if v is not None:
                    return v
            return ''

        header = f"-- Source: {source_label}\nPlan hash value: (stored plan)\n\n" if source_label else ''
        col_op, col_obj = 42, 22
        sep = '-' * (col_op + col_obj + 30)
        lines: List[str] = [header + sep]
        lines.append(f"| {'Id':>3} | {'Operation':<{col_op}} | {'Name':<{col_obj}} | {'Cost':>6} | {'Rows':>6} |")
        lines.append(sep)

        pred_sections: List[str] = []
        for r in rows:
            row_id = _gv(r, 'id') or 0
            depth  = int(_gv(r, 'depth') or 0)
            op     = str(_gv(r, 'operation') or '')
            opts   = str(_gv(r, 'options') or '')
            obj    = str(_gv(r, 'object_name') or '')
            cost_v = _gv(r, 'cost')
            card_v = _gv(r, 'cardinality')
            op_str = ('  ' * depth) + op + (' ' + opts if opts else '')
            cost_s = str(cost_v) if cost_v not in (None, '') else ''
            card_s = str(card_v) if card_v not in (None, '') else ''
            lines.append(
                f"| {row_id:>3} | {op_str:<{col_op}} | {obj:<{col_obj}} | {cost_s:>6} | {card_s:>6} |"
            )
            acc = str(_gv(r, 'access_predicates') or '')
            flt = str(_gv(r, 'filter_predicates') or '')
            if acc:
                pred_sections.append(f"   {row_id} - access({acc})")
            if flt:
                pred_sections.append(f"   {row_id} - filter({flt})")

        lines.append(sep)
        if pred_sections:
            lines.append('\nPredicate Information (identified by operation id):')
            lines.append('-' * 50)
            lines.extend(pred_sections)
        return '\n'.join(lines)

    def _parse_xplan_text_to_rows(self, plan_text: str) -> List[Dict]:
        """Parse DBMS_XPLAN.DISPLAY_CURSOR text into structured dicts compatible with
        v$sql_plan column names so the SQL-ID lookup frontend can render it.
        """
        import re
        rows: List[Dict] = []

        # ── collect predicates first ──────────────────────────────────────
        predicates: Dict[int, Dict[str, str]] = {}
        pred_m = re.search(
            r'Predicate Information[^\n]*\n[-\s]*\n(.*?)(?:\n\n|\Z)',
            plan_text, re.DOTALL | re.IGNORECASE
        )
        if pred_m:
            for pm in re.finditer(
                r'^\s*(\d+)\s*-\s*(access|filter)\((.+?)\)',
                pred_m.group(1), re.MULTILINE | re.IGNORECASE
            ):
                rid = int(pm.group(1))
                predicates.setdefault(rid, {})[pm.group(2).lower()] = pm.group(3)

        # ── parse plan rows  |[*] id | operation [indented] | object | … ──
        row_pat = re.compile(
            r'^\|[\s*]+([0-9]+)\s*\|\s*(.*?)\s*\|\s*([A-Za-z0-9_\$#"\.]*)',
            re.MULTILINE
        )
        # option suffixes that should be split from operation
        _OPT_SUFFIXES = (
            'BY LOCAL INDEX ROWID BATCHED', 'BY LOCAL INDEX ROWID',
            'BY INDEX ROWID BATCHED', 'BY INDEX ROWID',
            'SKIP SCAN', 'RANGE SCAN', 'UNIQUE SCAN',
            'FULL', 'DESCENDING',
        )
        for m in row_pat.finditer(plan_text):
            row_id  = int(m.group(1))
            op_raw  = m.group(2)
            obj     = m.group(3).strip().strip('"')
            depth   = (len(op_raw) - len(op_raw.lstrip())) // 2
            op_str  = op_raw.strip().upper()
            operation, options = op_str, ''
            for sfx in _OPT_SUFFIXES:
                if op_str.endswith(sfx):
                    operation = op_str[: -len(sfx)].strip()
                    options   = sfx
                    break
            preds = predicates.get(row_id, {})
            rows.append({
                'id': row_id,   'parent_id': None,
                'operation': operation, 'options': options,
                'object_name': obj,     'object_type': '',
                'cost': None,           'cardinality': None,
                'bytes': None,          'cpu_cost': None,
                'io_cost': None,        'temp_space': None,
                'access_predicates': preds.get('access', ''),
                'filter_predicates': preds.get('filter', ''),
                'depth': depth,
            })
        return rows

    def _analyze_plan_structure_foolproof(self, plan_rows: List[Dict]) -> Dict[str, Any]:
        """Analyze plan as STRUCTURED ROWS — foolproof vs Oracle version changes.
        
        No regex fragility. Uses v$sql_plan / awr_root_sql_plan columns directly.
        Returns specific issues with numeric evidence (costs, cardinality, etc.)
        """
        analysis = {
            'full_table_scans': [],
            'join_inefficiencies': [],
            'sort_operations': [],
            'missing_index_opportunities': [],
            'predicate_distribution': {},
        }
        
        if not plan_rows:
            return analysis

        # Build node lookup
        node_map = {}
        for row in plan_rows:
            try:
                node_id = int(row.get('id') or 0)
                node_map[node_id] = row
            except (ValueError, TypeError):
                continue

        for node_id, node in node_map.items():
            op = str(node.get('operation') or '').upper()
            opts = str(node.get('options') or '').upper()
            obj = node.get('object_name') or ''
            cost = _safe_float(node.get('cost') or 0)
            card = _safe_float(node.get('cardinality') or 0)
            access_pred = node.get('access_predicates') or ''
            filter_pred = node.get('filter_predicates') or ''

            # ─────── Full Table Scan detection ───────
            if 'TABLE ACCESS' in op and 'FULL' in opts:
                has_filter = bool(filter_pred)
                analysis['full_table_scans'].append({
                    'node_id': node_id,
                    'object': obj,  # anonymized at LLM stage
                    'cost': round(cost, 1),
                    'cardinality': _safe_int(card),
                    'has_filter': has_filter,
                    'filter_pred_exists': has_filter,
                    'access_pred_exists': bool(access_pred),
                    'recommendation': 'Create index' if has_filter else 'Verify necessity or gather stats',
                })

            # ─────── Join efficiency ───────
            if 'JOIN' in op or 'NESTED LOOPS' in op:
                # Flag NESTED LOOPS on large result sets
                if card > 50000 and 'NESTED LOOPS' in op:
                    analysis['join_inefficiencies'].append({
                        'node_id': node_id,
                        'type': 'NESTED LOOPS',
                        'joined_cardinality': _safe_int(card),
                        'cost': round(cost, 1),
                        'severity': 'HIGH' if card > 500000 else 'MEDIUM',
                        'recommendation': 'Consider HASH JOIN or improve join predicates',
                    })

            # ─────── Sort operations ───────
            if 'SORT' in op:
                bytes_val = _safe_float(node.get('bytes') or 0)
                analysis['sort_operations'].append({
                    'node_id': node_id,
                    'type': opts,
                    'cost': round(cost, 1),
                    'cardinality': _safe_int(card),
                    'bytes': _safe_int(bytes_val),
                    'sort_in_memory': bytes_val < 1e9,  # < 1GB typically in-memory
                })

            # ─────── Index opportunity detection ───────
            # If node has filter predicates AND is expensive, index on those columns would help
            if filter_pred and cost > 100 and 'TABLE ACCESS FULL' in op:
                # Extract column names from predicate (safely)
                pred_columns = len(re.findall(r'(?:=|>|<|>=|<=|LIKE|IN|BETWEEN)', filter_pred))
                if pred_columns > 0:
                    analysis['missing_index_opportunities'].append({
                        'node_id': node_id,
                        'operation': op,
                        'cost': round(cost, 1),
                        'filter_predicate_count': pred_columns,
                        'recommendation': f'Create index with {pred_columns} column(s) matching filter predicates',
                    })

        return analysis

    def _detect_cardinality_errors(self, display_cursor_plan: str, v_sql_plan_rows: List[Dict]) -> List[Dict]:
        """Detect where optimizer estimation is wildly off — structural error detection.
        
        Compares estimated vs actual rows. Returns specific error metrics without exposing table names.
        Example: node predicted 100 rows, actually got 100K = 1000x error (HIGH severity)
        """
        errors = []

        # Extract A-Rows (actual rows) from DISPLAY_CURSOR text if available
        a_rows_map = {}
        if display_cursor_plan:
            # Pattern: |  1 |  TABLE ACCESS FULL       |           ... |   1  |     1 |       100K (A-rows=1000)
            for m in re.finditer(r'\|\s*(\d+)\s*\|.*?A-Rows\s*=\s*([0-9\.KMG]+)', display_cursor_plan, re.IGNORECASE):
                try:
                    node_id = int(m.group(1))
                    a_rows_str = m.group(2).upper()
                    # Parse 100K, 1M, 1G format
                    a_rows = float(a_rows_str.rstrip('KMG'))
                    if 'K' in a_rows_str:
                        a_rows *= 1000
                    elif 'M' in a_rows_str:
                        a_rows *= 1e6
                    elif 'G' in a_rows_str:
                        a_rows *= 1e9
                    a_rows_map[node_id] = int(a_rows)
                except (ValueError, AttributeError):
                    continue

        # Compare estimated (E-Rows from v$sql_plan) vs actual (A-Rows from DISPLAY_CURSOR)
        for plan_row in v_sql_plan_rows:
            try:
                node_id = _safe_int(plan_row.get('id') or 0)
                e_rows = _safe_int(plan_row.get('cardinality') or 1, 1)
                a_rows = a_rows_map.get(node_id)

                if a_rows is None or a_rows == 0:
                    continue  # No actual row data available for this node

                if e_rows == 0:
                    e_rows = 1  # Avoid division by zero

                ratio = a_rows / e_rows

                # Flag significant mismatches
                if ratio > 100 or ratio < 0.01:  # 100x error either direction
                    op = plan_row.get('operation') or 'UNKNOWN'
                    severity = 'CRITICAL' if ratio > 1000 else 'CRITICAL' if ratio < 0.001 else 'HIGH'

                    errors.append({
                        'node_id': node_id,
                        'operation': op,
                        'estimated_rows': e_rows,
                        'actual_rows': a_rows,
                        'error_ratio': round(ratio, 2),
                        'severity': severity,
                        'recommendation': (
                            'Stale statistics — run DBMS_STATS.GATHER_TABLE_STATS with CASCADE=TRUE' 
                            if ratio > 1000 else
                            'Column correlation or missing statistics — analyze SELECT columns in WHERE clause'
                        ),
                        'impact': (
                            f'Optimizer may choose bad join orders or skip useful indexes (predicting {e_rows} vs actual {a_rows})'
                        ),
                    })
            except (ValueError, TypeError):
                continue

        return errors

    def _detect_bind_variable_skew(self, sql_id: str) -> Dict[str, Any]:
        """Detect if this SQL has different plans for different bind values (plan skew).
        
        Returns detection results without executing new queries — uses metrics already collected.
        """
        skew_info = {
            'has_skew': False,
            'child_cursor_count': 0,
            'plan_hash_variation': [],
            'recommendation': None,
        }

        try:
            # Query v$sql for same sql_id — look for multiple child cursors with different plans
            rows = self.conn.execute_query_dict(
                f"SELECT COUNT(DISTINCT plan_hash_value) AS plan_count, "
                f"       COUNT(DISTINCT child_number) AS child_count, "
                f"       SUM(executions) AS total_exec, "
                f"       COUNT(*) AS cursor_count "
                f"FROM v$sql WHERE sql_id = '{sql_id}'"
            )

            if rows and rows[0]:
                # Oracle returns uppercase column names; check both cases
                plan_count = int(rows[0].get('PLAN_COUNT') or rows[0].get('plan_count') or 1)
                child_count = int(rows[0].get('CHILD_COUNT') or rows[0].get('child_count') or 1)
                total_exec = int(rows[0].get('TOTAL_EXEC') or rows[0].get('total_exec') or 0)

                skew_info['child_cursor_count'] = child_count

                if plan_count > 1:
                    skew_info['has_skew'] = True
                    skew_info['plan_hash_variation'] = [{
                        'plan_variation_count': plan_count,
                        'total_child_cursors': child_count,
                        'total_executions': total_exec,
                        'severity': 'HIGH' if plan_count > 5 else 'MEDIUM',
                    }]
                    skew_info['recommendation'] = (
                        f'{plan_count} different plan_hash_values for same SQL_ID detected. '
                        'This indicates BIND PEEKING variation. '
                        'Consider: (1) Add BIND_AWARE hints, (2) Use CURSOR_SHARING=EXACT, (3) Compile with _bind_aware=FALSE'
                    )
        except Exception as ex:
            logger.warning(f"Bind skew detection failed: {ex}")

        return skew_info

    def _generate_specific_recommendations(
        self,
        plan_analysis: Dict[str, Any],
        cardinality_errors: List[Dict],
        bind_skew: Dict[str, Any],
        exec_stats: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Generate SPECIFIC (not generic) recommendations from structured analysis.
        
        Each recommendation is tied to numeric evidence from the database.
        No table names or actual data exposed — only metrics, costs, cardinality.
        """
        recommendations = []

        # ─────── Full Table Scan opportunities ───────
        for fts in plan_analysis.get('full_table_scans', []):
            if fts['has_filter'] and fts['cardinality'] > 100:
                recommendations.append({
                    'type': 'INDEX',
                    'severity': 'HIGH' if fts['cardinality'] > 1000000 else 'MEDIUM',
                    'title': 'Create index on filtered column(s)',
                    'evidence': {
                        'node_id': fts['node_id'],
                        'full_table_scan_cardinality': fts['cardinality'],
                        'estimated_cost': fts['cost'],
                        'filter_present': fts['has_filter'],
                    },
                    'expected_benefit': f"Eliminate full table scan; reduce from cost={fts['cost']:.1f} to ~10-20 with index access",
                    'risk_level': 'LOW',
                    'action': 'Identify filter columns from plan and create composite index',
                })

        # ─────── Cardinality errors → Statistics gathering ───────
        high_error_count = len([e for e in cardinality_errors if e['severity'] == 'CRITICAL'])
        if high_error_count > 0:
            avg_ratio = sum(e['error_ratio'] for e in cardinality_errors) / len(cardinality_errors)
            recommendations.append({
                'type': 'STATISTICS',
                'severity': 'CRITICAL',
                'title': f'Gather stale statistics — {high_error_count} critical cardinality errors detected',
                'evidence': {
                    'error_count': len(cardinality_errors),
                    'critical_errors': high_error_count,
                    'average_error_ratio': round(avg_ratio, 2),
                    'largest_error': max((e['error_ratio'] for e in cardinality_errors), default=0),
                },
                'expected_benefit': (
                    'Optimizer will make better plan choices. Expected latency improvement: 20-70% '
                    'depending on which tables have stale statistics'
                ),
                'risk_level': 'VERY_LOW',
                'action': 'Run DBMS_STATS.GATHER_TABLE_STATS with ESTIMATE_PERCENT=AUTO for affected tables',
            })

        # ─────── Join inefficiency ───────
        for ji in plan_analysis.get('join_inefficiencies', []):
            recommendations.append({
                'type': 'JOIN_OPTIMIZATION',
                'severity': ji['severity'],
                'title': f"NESTED LOOPS join on {ji['joined_cardinality']:,} rows — consider HASH JOIN",
                'evidence': {
                    'node_id': ji['node_id'],
                    'joined_rows': ji['joined_cardinality'],
                    'nested_loop_cost': ji['cost'],
                    'estimated_hash_cost_improvement': f"{max(5, ji['cost'] // 2)}% reduction",
                },
                'expected_benefit': 'Reduce join cost by 30-80% for large result sets',
                'risk_level': 'LOW',
                'action': 'Add /*+ USE_HASH(t1, t2) */ hint or improve join predicates to enable filter-based optimization',
            })

        # ─────── Sort spill warnings ───────
        for sort in plan_analysis.get('sort_operations', []):
            if not sort['sort_in_memory']:
                size_gb = sort['bytes'] / 1e9
                recommendations.append({
                    'type': 'MEMORY_TUNING',
                    'severity': 'MEDIUM',
                    'title': f'Sort operation spilling to disk ({size_gb:.1f} GB)',
                    'evidence': {
                        'node_id': sort['node_id'],
                        'sort_bytes': sort['bytes'],
                        'sort_cost': sort['cost'],
                        'cardinality': sort['cardinality'],
                    },
                    'expected_benefit': 'Reduce sort time by 50-90% by keeping sort in memory',
                    'risk_level': 'LOW',
                    'action': 'Increase SORT_AREA_SIZE / PGA_AGGREGATE_TARGET or add ORDER BY index',
                })

        # ─────── Bind variable skew ───────
        if bind_skew.get('has_skew'):
            recommendations.append({
                'type': 'BIND_SKEW',
                'severity': 'HIGH',
                'title': 'Bind variable peeking causing plan variation',
                'evidence': bind_skew.get('plan_hash_variation', [{}])[0],
                'expected_benefit': 'Consistent plan selection; prevent periodic performance drop when bind values change',
                'risk_level': 'MEDIUM',
                'action': (
                    'Enable Adaptive Cursor Sharing (ALTER SESSION SET "_optimizer_adaptive_cursor_sharing" = TRUE), '
                    'or create SQL Plan Baselines via DBMS_SPM to stabilize the optimal plan'
                ),
            })

        # ─────── Sort operations ───────
        if plan_analysis.get('missing_index_opportunities'):
            recommendations.append({
                'type': 'MISSING_INDEX',
                'severity': 'MEDIUM',
                'title': f"{len(plan_analysis['missing_index_opportunities'])} index opportunities on filtered columns",
                'evidence': {
                    'opportunity_count': len(plan_analysis['missing_index_opportunities']),
                    'total_potential_cost_reduction': sum(
                        x.get('cost', 0) for x in plan_analysis['missing_index_opportunities']
                    ),
                },
                'expected_benefit': 'Reduce query cost and latency by 40-70%',
                'risk_level': 'LOW',
                'action': 'Create indexes on leading predicates identified in plan',
            })

        return recommendations

    def _collect_object_stats_for_plan(
        self,
        plan_rows: List[Dict[str, Any]],
        predicate_columns: List[str],
    ) -> Dict[str, Any]:
        """Collect table, column, and index statistics for objects in the execution plan.

        Uses (owner, table_name) pairs extracted from plan rows when object_owner is
        available (v$sql_plan / awr_root_sql_plan), which gives accurate per-schema
        filtering.  Falls back to table_name-only matching when owner is absent
        (DBMS_XPLAN-parsed rows).

        Tries dba_tab_statistics / dba_tab_col_statistics / dba_indexes first
        (requires SELECT_CATALOG_ROLE or DBA).  If those return nothing — possible
        privilege gap — retries with all_tab_statistics / all_tab_col_statistics /
        all_indexes, which are accessible to any user.
        """
        if not plan_rows:
            return {
                'table_stats': [],
                'column_stats': [],
                'index_stats': [],
                'table_partition_stats': [],
                'index_partition_stats': [],
            }

        # Extract unique (owner, table_name) and index name from plan rows.
        # object_owner is available when plan_rows come from v$sql_plan or
        # awr_root_sql_plan (we now SELECT it). For DBMS_XPLAN-parsed rows it is
        # absent; those cases fall back to table-name-only matching.
        owner_table_pairs: List[tuple] = []   # (owner, table_name) — owner may be ''
        table_names: List[str] = []
        index_names: List[str] = []
        seen_tables: set = set()
        seen_indexes: set = set()
        for row in plan_rows:
            if not isinstance(row, dict):
                continue
            op  = str(row.get('operation')    or row.get('OPERATION')    or '').upper()
            obj = str(row.get('object_name')  or row.get('OBJECT_NAME')  or '').strip().upper()
            own = str(row.get('object_owner') or row.get('OBJECT_OWNER') or '').strip().upper()
            if not obj:
                continue
            safe_obj = ''.join(ch for ch in obj if ch.isalnum() or ch in ('_', '$', '#'))
            safe_own = ''.join(ch for ch in own if ch.isalnum() or ch in ('_', '$', '#'))
            if not safe_obj:
                continue
            if 'TABLE' in op:
                if safe_obj not in seen_tables:
                    seen_tables.add(safe_obj)
                    table_names.append(safe_obj)
                    owner_table_pairs.append((safe_own, safe_obj))
            elif 'INDEX' in op:
                if safe_obj not in seen_indexes:
                    seen_indexes.add(safe_obj)
                    index_names.append(safe_obj)

        if not table_names and not index_names:
            return {
                'table_stats': [],
                'column_stats': [],
                'index_stats': [],
                'table_partition_stats': [],
                'index_partition_stats': [],
            }

        safe_tables  = table_names[:20]
        safe_indexes = index_names[:20]

        # Build owner-aware filter: WHERE (owner, table_name) IN (...)
        # Falls back to table_name-only when no owner info is available.
        def _owner_table_filter(alias: str, pairs: List[tuple]) -> str:
            """Return a SQL WHERE fragment that filters by (owner, table_name) when
            owners are known, or table_name IN (...) when they are not."""
            known_pairs = [(o, t) for o, t in pairs if o]
            all_tables  = [t for _, t in pairs]
            if known_pairs:
                pair_list = ', '.join(f"('{o}', '{t}')" for o, t in known_pairs)
                return f"({alias}.owner, {alias}.table_name) IN ({pair_list})"
            tbl_in = ', '.join(f"'{t}'" for t in all_tables)
            return f"{alias}.table_name IN ({tbl_in})"

        table_stats = []
        column_stats = []
        index_stats = []
        table_partition_stats = []
        index_partition_stats = []

        # Build batch queries for stats collection
        stats_queries: Dict[str, str] = {}

        if safe_tables:
            tbl_filter  = _owner_table_filter('ts', owner_table_pairs)
            tbl_filter_i = _owner_table_filter('i', owner_table_pairs)
            tbl_in = ', '.join(f"'{t}'" for t in safe_tables)

            stats_queries['table_stats'] = (
                f"SELECT ts.owner, ts.table_name, ts.num_rows, ts.blocks, ts.avg_row_len, "
                f"TO_CHAR(ts.last_analyzed, 'YYYY-MM-DD HH24:MI:SS') AS last_analyzed, "
                f"ts.stale_stats, ts.partitioned, ts.degree, ts.sample_size "
                f"FROM dba_tab_statistics ts "
                f"WHERE {tbl_filter} "
                f"AND ts.object_type = 'TABLE' "
                f"ORDER BY ts.owner, ts.table_name"
            )

            stats_queries['table_partition_stats'] = (
                f"SELECT ts.owner, ts.table_name, ts.partition_name, ts.subpartition_name, ts.object_type, "
                f"ts.num_rows, ts.blocks, ts.sample_size, ts.stale_stats, "
                f"TO_CHAR(ts.last_analyzed, 'YYYY-MM-DD HH24:MI:SS') AS last_analyzed "
                f"FROM dba_tab_statistics ts "
                f"WHERE {tbl_filter} "
                f"AND ts.object_type IN ('PARTITION', 'SUBPARTITION') "
                f"ORDER BY ts.owner, ts.table_name, ts.partition_name, ts.subpartition_name "
                f"FETCH FIRST 500 ROWS ONLY"
            )

            # Column stats for plan-used predicate columns + index columns
            safe_pred_cols = [''.join(ch for ch in c if ch.isalnum() or ch in ('_', '$', '#'))
                             for c in (predicate_columns or [])[:60]]
            safe_pred_cols = [c for c in safe_pred_cols if c]
            if safe_pred_cols or safe_indexes:
                tbl_filter_cs = _owner_table_filter('cs', owner_table_pairs)
                col_filters = []
                if safe_pred_cols:
                    col_in = ', '.join(f"'{c}'" for c in safe_pred_cols)
                    col_filters.append(
                        f"({tbl_filter_cs} AND cs.column_name IN ({col_in}))"
                    )
                if safe_indexes:
                    idx_in = ', '.join(f"'{i}'" for i in safe_indexes)
                    col_filters.append(
                        f"({tbl_filter_cs} AND (cs.table_name, cs.column_name) IN ("
                        f"SELECT ic.table_name, ic.column_name FROM dba_ind_columns ic "
                        f"WHERE ic.index_name IN ({idx_in})"
                        f"))"
                    )
                if col_filters:
                    stats_queries['column_stats'] = (
                        f"SELECT cs.owner, cs.table_name, cs.column_name, cs.num_distinct, cs.num_nulls, "
                        f"cs.density, cs.low_value, cs.high_value, cs.histogram, cs.num_buckets, "
                        f"TO_CHAR(cs.last_analyzed, 'YYYY-MM-DD HH24:MI:SS') AS last_analyzed, "
                        f"cs.sample_size, cs.avg_col_len "
                        f"FROM dba_tab_col_statistics cs "
                        f"WHERE {' OR '.join(col_filters)} "
                        f"ORDER BY cs.owner, cs.table_name, cs.column_name"
                    )

        # Index stats for all indexes on referenced tables
        if safe_tables:
            stats_queries['index_stats'] = (
                f"SELECT i.owner, i.table_name, i.index_name, i.index_type, "
                f"i.uniqueness, i.status, i.num_rows, i.distinct_keys, "
                f"i.clustering_factor, i.blevel, i.leaf_blocks, "
                f"TO_CHAR(i.last_analyzed, 'YYYY-MM-DD HH24:MI:SS') AS last_analyzed, "
                f"i.degree, i.partitioned, i.visibility, "
                f"LISTAGG(c.column_name, ', ') WITHIN GROUP (ORDER BY c.column_position) AS index_columns "
                f"FROM dba_indexes i "
                f"LEFT JOIN dba_ind_columns c "
                f"  ON c.index_owner = i.owner AND c.index_name = i.index_name "
                f"WHERE {tbl_filter_i} "
                f"GROUP BY i.owner, i.table_name, i.index_name, i.index_type, "
                f"i.uniqueness, i.status, i.num_rows, i.distinct_keys, "
                f"i.clustering_factor, i.blevel, i.leaf_blocks, "
                f"i.last_analyzed, i.degree, i.partitioned, i.visibility "
                f"ORDER BY i.owner, i.table_name, i.index_name"
            )

        if safe_indexes:
            idx_in = ', '.join(f"'{i}'" for i in safe_indexes)
            stats_queries['index_partition_stats'] = (
                f"SELECT owner, index_name, partition_name, subpartition_name, object_type, "
                f"blevel, leaf_blocks, distinct_keys, clustering_factor, num_rows, sample_size, stale_stats, "
                f"TO_CHAR(last_analyzed, 'YYYY-MM-DD HH24:MI:SS') AS last_analyzed "
                f"FROM dba_ind_statistics "
                f"WHERE index_name IN ({idx_in}) "
                f"AND object_type IN ('PARTITION', 'SUBPARTITION') "
                f"ORDER BY index_name, partition_name, subpartition_name "
                f"FETCH FIRST 500 ROWS ONLY"
            )

        if not stats_queries:
            return {
                'table_stats': [],
                'column_stats': [],
                'index_stats': [],
                'table_partition_stats': [],
                'index_partition_stats': [],
            }

        # ── Execute — dba_* views first, fall back to all_* on empty/privilege failure ──
        def _run_batch(queries):
            try:
                if hasattr(self.conn, 'execute_batch_queries_dict'):
                    return self.conn.execute_batch_queries_dict(queries)
                r = {}
                for lbl, sql in queries.items():
                    try:
                        r[lbl] = self.conn.execute_query_dict(sql) or []
                    except Exception:
                        r[lbl] = []
                return r
            except Exception:
                return {lbl: [] for lbl in queries}

        results = _run_batch(stats_queries)

        # Fall back to all_* views for any category that came back empty.
        # all_tab_statistics / all_tab_col_statistics / all_indexes are accessible
        # without SELECT_CATALOG_ROLE, so this covers non-DBA users.
        fallback_needed = {
            k for k in ('table_stats', 'column_stats', 'index_stats',
                         'table_partition_stats', 'index_partition_stats')
            if not results.get(k)
        }
        if fallback_needed:
            fallback_queries: Dict[str, str] = {}
            for k in fallback_needed:
                if k in stats_queries:
                    fallback_queries[k] = (
                        stats_queries[k]
                        .replace('dba_tab_statistics', 'all_tab_statistics')
                        .replace('dba_tab_col_statistics', 'all_tab_col_statistics')
                        .replace('dba_indexes', 'all_indexes')
                        .replace('dba_ind_columns', 'all_ind_columns')
                        .replace('dba_ind_statistics', 'all_ind_statistics')
                    )
            if fallback_queries:
                fb_results = _run_batch(fallback_queries)
                for k, rows in fb_results.items():
                    if rows:
                        results[k] = rows
                        logger.info("Object stats: '%s' filled from all_* fallback (%d rows)", k, len(rows))

        # Process table stats
        for r in results.get('table_stats', []):
            table_stats.append({
                'owner': r.get('OWNER') or r.get('owner'),
                'table_name': r.get('TABLE_NAME') or r.get('table_name'),
                'num_rows': r.get('NUM_ROWS') or r.get('num_rows'),
                'blocks': r.get('BLOCKS') or r.get('blocks'),
                'avg_row_len': r.get('AVG_ROW_LEN') or r.get('avg_row_len'),
                'last_analyzed': r.get('LAST_ANALYZED') or r.get('last_analyzed'),
                'stale_stats': r.get('STALE_STATS') or r.get('stale_stats'),
                'partitioned': r.get('PARTITIONED') or r.get('partitioned'),
                'degree': r.get('DEGREE') or r.get('degree'),
                'sample_size': r.get('SAMPLE_SIZE') or r.get('sample_size'),
            })

        # Process column stats
        for r in results.get('column_stats', []):
            column_stats.append({
                'owner': r.get('OWNER') or r.get('owner'),
                'table_name': r.get('TABLE_NAME') or r.get('table_name'),
                'column_name': r.get('COLUMN_NAME') or r.get('column_name'),
                'num_distinct': r.get('NUM_DISTINCT') or r.get('num_distinct'),
                'num_nulls': r.get('NUM_NULLS') or r.get('num_nulls'),
                'density': r.get('DENSITY') or r.get('density'),
                'histogram': r.get('HISTOGRAM') or r.get('histogram'),
                'num_buckets': r.get('NUM_BUCKETS') or r.get('num_buckets'),
                'last_analyzed': r.get('LAST_ANALYZED') or r.get('last_analyzed'),
                'sample_size': r.get('SAMPLE_SIZE') or r.get('sample_size'),
                'avg_col_len': r.get('AVG_COL_LEN') or r.get('avg_col_len'),
            })

        # Process index stats
        for r in results.get('index_stats', []):
            index_stats.append({
                'owner': r.get('OWNER') or r.get('owner'),
                'table_name': r.get('TABLE_NAME') or r.get('table_name'),
                'index_name': r.get('INDEX_NAME') or r.get('index_name'),
                'index_type': r.get('INDEX_TYPE') or r.get('index_type'),
                'uniqueness': r.get('UNIQUENESS') or r.get('uniqueness'),
                'status': r.get('STATUS') or r.get('status'),
                'num_rows': r.get('NUM_ROWS') or r.get('num_rows'),
                'distinct_keys': r.get('DISTINCT_KEYS') or r.get('distinct_keys'),
                'clustering_factor': r.get('CLUSTERING_FACTOR') or r.get('clustering_factor'),
                'blevel': r.get('BLEVEL') or r.get('blevel'),
                'leaf_blocks': r.get('LEAF_BLOCKS') or r.get('leaf_blocks'),
                'last_analyzed': r.get('LAST_ANALYZED') or r.get('last_analyzed'),
                'degree': r.get('DEGREE') or r.get('degree'),
                'partitioned': r.get('PARTITIONED') or r.get('partitioned'),
                'visibility': r.get('VISIBILITY') or r.get('visibility'),
                'index_columns': r.get('INDEX_COLUMNS') or r.get('index_columns') or '',
            })

        for r in results.get('table_partition_stats', []):
            table_partition_stats.append({
                'owner': r.get('OWNER') or r.get('owner'),
                'table_name': r.get('TABLE_NAME') or r.get('table_name'),
                'partition_name': r.get('PARTITION_NAME') or r.get('partition_name'),
                'subpartition_name': r.get('SUBPARTITION_NAME') or r.get('subpartition_name'),
                'object_type': r.get('OBJECT_TYPE') or r.get('object_type'),
                'num_rows': r.get('NUM_ROWS') or r.get('num_rows'),
                'blocks': r.get('BLOCKS') or r.get('blocks'),
                'sample_size': r.get('SAMPLE_SIZE') or r.get('sample_size'),
                'stale_stats': r.get('STALE_STATS') or r.get('stale_stats'),
                'last_analyzed': r.get('LAST_ANALYZED') or r.get('last_analyzed'),
            })

        for r in results.get('index_partition_stats', []):
            index_partition_stats.append({
                'owner': r.get('OWNER') or r.get('owner'),
                'index_name': r.get('INDEX_NAME') or r.get('index_name'),
                'partition_name': r.get('PARTITION_NAME') or r.get('partition_name'),
                'subpartition_name': r.get('SUBPARTITION_NAME') or r.get('subpartition_name'),
                'object_type': r.get('OBJECT_TYPE') or r.get('object_type'),
                'blevel': r.get('BLEVEL') or r.get('blevel'),
                'leaf_blocks': r.get('LEAF_BLOCKS') or r.get('leaf_blocks'),
                'distinct_keys': r.get('DISTINCT_KEYS') or r.get('distinct_keys'),
                'clustering_factor': r.get('CLUSTERING_FACTOR') or r.get('clustering_factor'),
                'num_rows': r.get('NUM_ROWS') or r.get('num_rows'),
                'sample_size': r.get('SAMPLE_SIZE') or r.get('sample_size'),
                'stale_stats': r.get('STALE_STATS') or r.get('stale_stats'),
                'last_analyzed': r.get('LAST_ANALYZED') or r.get('last_analyzed'),
            })

        return {
            'table_stats': table_stats,
            'column_stats': column_stats,
            'index_stats': index_stats,
            'table_partition_stats': table_partition_stats,
            'index_partition_stats': index_partition_stats,
        }

    def _collect_existing_indexes_for_plan(
        self,
        plan_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Collect existing indexes for tables referenced in the execution plan."""
        if not plan_rows:
            return []

        table_names: List[str] = []
        seen = set()
        for row in plan_rows:
            if not isinstance(row, dict):
                continue
            op = str(row.get('operation') or row.get('OPERATION') or '').upper()
            obj = str(row.get('object_name') or row.get('OBJECT_NAME') or '').strip().upper()
            if not obj:
                continue
            if 'TABLE ACCESS' in op or 'INDEX' in op:
                if obj not in seen:
                    seen.add(obj)
                    table_names.append(obj)

        if not table_names:
            return []

        safe_tables = []
        for t in table_names[:20]:
            safe_t = ''.join(ch for ch in t if ch.isalnum() or ch in ('_', '$', '#'))
            if safe_t:
                safe_tables.append(safe_t)
        if not safe_tables:
            return []

        in_clause = ', '.join(f"'{t}'" for t in safe_tables)

        try:
            rows = self.conn.execute_query_dict(
                "SELECT i.owner, i.table_name, i.index_name, i.uniqueness, i.status, "
                "       LISTAGG(c.column_name, ', ') WITHIN GROUP (ORDER BY c.column_position) AS index_columns "
                "FROM dba_indexes i "
                "LEFT JOIN dba_ind_columns c "
                "  ON c.index_owner = i.owner AND c.index_name = i.index_name "
                f"WHERE i.table_name IN ({in_clause}) "
                "GROUP BY i.owner, i.table_name, i.index_name, i.uniqueness, i.status "
                "ORDER BY i.table_name, i.index_name"
            ) or []
        except Exception:
            return []

        result: List[Dict[str, Any]] = []
        for r in rows:
            result.append({
                'owner': r.get('OWNER') or r.get('owner'),
                'table_name': r.get('TABLE_NAME') or r.get('table_name'),
                'index_name': r.get('INDEX_NAME') or r.get('index_name'),
                'uniqueness': r.get('UNIQUENESS') or r.get('uniqueness'),
                'status': r.get('STATUS') or r.get('status'),
                'index_columns': r.get('INDEX_COLUMNS') or r.get('index_columns') or '',
            })
        return result

    def _get_plan_for_llm_anonymized(
        self,
        sql_analysis_entry: Dict[str, Any],
        cardinality_errors: List[Dict],
        bind_skew: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare detailed plan context for LLM with table and predicate visibility."""
        plan_analysis = sql_analysis_entry.get('plan_analysis', {}) or {}

        # ── Extract SQL text ──
        # Supports both collect_sqlid_info format (list of dicts) and
        # collect_top_sql_with_plans format (query_text string).
        sql_text_full = ''
        sql_text_rows = sql_analysis_entry.get('sql_text', [])
        if sql_text_rows and isinstance(sql_text_rows, list) and sql_text_rows:
            row0 = sql_text_rows[0] if sql_text_rows else {}
            sql_text_full = str(
                row0.get('SQL_FULLTEXT') or row0.get('sql_fulltext') or
                row0.get('SQL_TEXT') or row0.get('sql_text') or ''
            )[:4000]  # cap at 4000 chars for LLM token budget
        if not sql_text_full:
            # Fallback: collect_top_sql_with_plans stores text in 'query_text'
            sql_text_full = str(sql_analysis_entry.get('query_text') or '')[:4000]

        # ── Extract raw DBMS_XPLAN text (has actual row counts) ──
        raw_plan_text = str(sql_analysis_entry.get('execution_plan_text') or '')[:6000]
        if not raw_plan_text:
            # Fallback: collect_top_sql_with_plans stores plan text in 'execution_plan'
            ep = sql_analysis_entry.get('execution_plan')
            if isinstance(ep, str):
                raw_plan_text = ep[:6000]

        # Prefer structured plan rows when available.
        plan_rows = []
        exec_plan_rows = sql_analysis_entry.get('execution_plan_rows', [])
        exec_plan = sql_analysis_entry.get('execution_plan', [])
        if isinstance(exec_plan_rows, list) and exec_plan_rows:
            plan_rows = exec_plan_rows
        elif isinstance(exec_plan, list) and exec_plan:
            plan_rows = exec_plan

        operations = []
        for row in plan_rows[:30]:
            if not isinstance(row, dict):
                continue
            try:
                operations.append({
                    'id': row.get('id') or row.get('ID'),
                    'parent_id': row.get('parent_id') or row.get('PARENT_ID'),
                    'operation': row.get('operation') or row.get('OPERATION'),
                    'options': row.get('options') or row.get('OPTIONS'),
                    'object_name': row.get('object_name') or row.get('OBJECT_NAME'),
                    'cost': _safe_float(row.get('cost') or row.get('COST') or 0),
                    'cardinality': _safe_int(row.get('cardinality') or row.get('CARDINALITY') or 0),
                    'access_predicates': row.get('access_predicates') or row.get('ACCESS_PREDICATES') or None,
                    'filter_predicates': row.get('filter_predicates') or row.get('FILTER_PREDICATES') or None,
                })
            except (TypeError, ValueError):
                continue

        # Fallback: derive minimal operations from analysis if row-level plan unavailable.
        if not operations:
            for fts in plan_analysis.get('full_table_scans', [])[:20]:
                operations.append({
                    'id': fts.get('node_id'),
                    'operation': 'TABLE ACCESS FULL',
                    'object_name': fts.get('object'),
                    'cost': fts.get('cost', 0),
                    'cardinality': fts.get('cardinality', 0),
                    'access_predicates': '(not available)',
                    'filter_predicates': '(not available)',
                })

        existing_indexes = self._collect_existing_indexes_for_plan(plan_rows)
        predicate_columns = []
        pred_regex = re.compile(r'([A-Z][A-Z0-9_#$]{1,30})\s*(?:=|<|>|<=|>=|LIKE|IN|BETWEEN)', re.IGNORECASE)
        for op in operations:
            for pred in (op.get('access_predicates'), op.get('filter_predicates')):
                if not pred:
                    continue
                for m in pred_regex.finditer(str(pred)):
                    col = m.group(1).upper()
                    if col not in predicate_columns:
                        predicate_columns.append(col)

        # ── Build current plan assessment for accurate LLM guidance ──
        all_plans_list = sql_analysis_entry.get('all_plans') or []
        exec_stats_raw = sql_analysis_entry.get('execution_stats') or []
        # If no per-plan execution_stats (e.g. from collect_top_sql_with_plans),
        # synthesize from top-level entry fields for assessment and LLM context.
        if not exec_stats_raw and sql_analysis_entry.get('executions'):
            exec_stats_raw = [{
                'PLAN_HASH_VALUE': sql_analysis_entry.get('current_plan_hash_value', ''),
                'EXECUTIONS': sql_analysis_entry.get('executions', 0),
                'AVG_ELAPSED_MS': sql_analysis_entry.get('avg_elapsed_ms', 0),
                'AVG_BUFFER_GETS': (
                    round(sql_analysis_entry.get('buffer_gets', 0) /
                          max(sql_analysis_entry.get('executions', 1), 1), 2)
                    if sql_analysis_entry.get('buffer_gets') else 0
                ),
            }]
        current_plan_assessment = self._assess_current_plan(
            all_plans_list, operations, exec_stats_raw
        )

        # ── Reuse already-collected object statistics from the SQL-ID data ──
        # sql_analysis_entry has 'object_statistics' populated by _collect_oracle_sqlid_info.
        # Reusing it avoids a second round-trip and guarantees the LLM sees exactly the
        # same data that the UI renders.
        object_stats = sql_analysis_entry.get('object_statistics') or {}
        if not object_stats or not any(
            object_stats.get(k) for k in ('table_stats', 'column_stats', 'index_stats')
        ):
            # Only fall back to a fresh collection if the cached data is genuinely absent.
            object_stats = self._collect_object_stats_for_plan(plan_rows, predicate_columns)

        return {
            'sql_id': sql_analysis_entry.get('sql_id'),
            'sql_text': sql_text_full,
            'execution_plan_text': raw_plan_text,
            'plan_source': sql_analysis_entry.get('plan_source', 'DISPLAY_CURSOR'),
            'plan_hash': (
                sql_analysis_entry.get('current_plan_hash_value')
                or (sql_analysis_entry.get('plan_comparison') or {}).get('current_plan_hash_value')
                or sql_analysis_entry.get('plan_hash_value')
            ),
            'operations': operations,
            'execution_stats': [
                {
                    'child_number': _safe_int(s.get('CHILD_NUMBER') or s.get('child_number') or 0),
                    'plan_hash_value': str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or ''),
                    'executions': _safe_int(s.get('EXECUTIONS') or s.get('executions') or 0),
                    'avg_elapsed_ms': _safe_float(s.get('AVG_ELAPSED_MS') or s.get('avg_elapsed_ms') or 0),
                    'avg_cpu_ms': _safe_float(s.get('AVG_CPU_MS') or s.get('avg_cpu_ms') or 0),
                    'avg_buffer_gets': _safe_float(s.get('AVG_BUFFER_GETS') or s.get('avg_buffer_gets') or 0),
                    'avg_disk_reads': _safe_float(s.get('AVG_DISK_READS') or s.get('avg_disk_reads') or 0),
                    'avg_rows': _safe_float(s.get('AVG_ROWS') or s.get('avg_rows') or 0),
                }
                for s in (exec_stats_raw or [])[:20]
            ],
            'execution_history_daily': [
                {
                    'exec_date': str(s.get('EXEC_DATE') or s.get('exec_date') or ''),
                    'plan_hash_value': str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or ''),
                    'inst_id': _safe_int(s.get('INST_ID') or s.get('inst_id') or 0),
                    'module': str(s.get('MODULE') or s.get('module') or ''),
                    'execs': _safe_int(s.get('EXECS') or s.get('execs') or 0),
                    'buffer_gets': _safe_float(s.get('BUFFER_GETS') or s.get('buffer_gets') or 0),
                    'rows_processed': _safe_float(s.get('ROWS_PROCESSED') or s.get('rows_processed') or 0),
                    'cpu_tim_secs': _safe_float(s.get('CPU_TIM_SECS') or s.get('cpu_tim_secs') or 0),
                    'ela_tim_secs': _safe_float(s.get('ELA_TIM_SECS') or s.get('ela_tim_secs') or 0),
                    'avg_ela_secs': _safe_float(s.get('AVG_ELA_SECS') or s.get('avg_ela_secs') or 0),
                    'px_tot': _safe_int(s.get('PX_TOT') or s.get('px_tot') or 0),
                }
                for s in (sql_analysis_entry.get('execution_history_daily') or [])[:40]
                if isinstance(s, dict)
            ],
            'available_plan_hash_values': [
                str(v) for v in (sql_analysis_entry.get('available_plan_hash_values') or [])[:50]
            ],
            'all_plans': [
                {
                    'plan_hash_value': p.get('plan_hash_value'),
                    'child_number': p.get('child_number'),
                    'exec_stats': p.get('exec_stats'),
                    'plan_operations': [
                        {
                            'id': r.get('id') or r.get('ID'),
                            'operation': r.get('operation') or r.get('OPERATION'),
                            'options': r.get('options') or r.get('OPTIONS'),
                            'object_name': r.get('object_name') or r.get('OBJECT_NAME'),
                            'cost': _safe_float(r.get('cost') or r.get('COST') or 0),
                            'cardinality': _safe_int(r.get('cardinality') or r.get('CARDINALITY') or 0),
                            'access_predicates': r.get('access_predicates') or r.get('ACCESS_PREDICATES') or None,
                            'filter_predicates': r.get('filter_predicates') or r.get('FILTER_PREDICATES') or None,
                        }
                        for r in (p.get('plan_rows') or [])[:30]
                        if isinstance(r, dict)
                    ],
                }
                for p in (sql_analysis_entry.get('all_plans') or [])[:10]
            ],
            'cardinality_errors': [
                {
                    'node_id': e.get('node_id'),
                    'operation': e.get('operation'),
                    'estimated_rows': e.get('estimated_rows'),
                    'actual_rows': e.get('actual_rows'),
                    'error_ratio': e.get('error_ratio'),
                    'severity': e.get('severity'),
                }
                for e in (cardinality_errors or [])[:5]  # top 5 errors only
            ],
            'bind_variable_skew': {
                'detected': bool(bind_skew.get('has_skew', False)),
                'plan_variations': bind_skew.get('plan_hash_variation', []),
            },
            'existing_indexes': existing_indexes,
            'predicate_columns_detected': predicate_columns,
            'current_plan_assessment': current_plan_assessment,
            'object_statistics': object_stats,
            'alternate_plans': sql_analysis_entry.get('alternate_plans', []),
            'plan_comparison': sql_analysis_entry.get('plan_comparison', {}),
            'specific_recommendations': sql_analysis_entry.get('specific_recommendations', []),
            'active_sessions': sql_analysis_entry.get('active_sessions', []),
            'session_wait_details': sql_analysis_entry.get('session_wait_details', []),
            'session_statistics': sql_analysis_entry.get('session_statistics', []),
            'transaction_rollback_info': sql_analysis_entry.get('transaction_rollback_info', []),
            'sort_usage_info': sql_analysis_entry.get('sort_usage_info', []),
        }

    def _assess_current_plan(
        self,
        all_plans: List[Dict[str, Any]],
        primary_operations: List[Dict[str, Any]],
        execution_stats: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build an assessment of the current/primary execution plan for LLM guidance."""
        # Identify the current plan = highest executions
        current_plan_rows = primary_operations
        current_phv = ''
        current_execs = 0

        if all_plans:
            best = all_plans[0]
            for p in all_plans:
                p_execs = _safe_int((p.get('exec_stats') or {}).get('executions') or
                                    (p.get('exec_stats') or {}).get('EXECUTIONS') or 0)
                b_execs = _safe_int((best.get('exec_stats') or {}).get('executions') or
                                    (best.get('exec_stats') or {}).get('EXECUTIONS') or 0)
                if p_execs > b_execs:
                    best = p
            current_plan_rows = best.get('plan_rows') or []
            current_phv = str(best.get('plan_hash_value') or '')
            current_execs = _safe_int((best.get('exec_stats') or {}).get('executions') or 0)

        if not current_plan_rows:
            current_plan_rows = primary_operations

        # Analyze operations in current plan
        uses_index = False
        has_full_table_scan = False
        index_operations = []
        fts_tables = []

        for row in current_plan_rows:
            if not isinstance(row, dict):
                continue
            op = str(row.get('operation') or row.get('OPERATION') or '').upper()
            opts = str(row.get('options') or row.get('OPTIONS') or '').upper()
            obj = str(row.get('object_name') or row.get('OBJECT_NAME') or '')
            if 'INDEX' in op:
                uses_index = True
                index_operations.append(f"{op} {opts} on {obj}".strip())
            if 'TABLE' in op and 'FULL' in opts:
                has_full_table_scan = True
                fts_tables.append(obj)

        # Determine plan quality
        plan_is_optimal = uses_index and not has_full_table_scan
        needs_optimization = has_full_table_scan and not uses_index

        # Get perf stats for current plan
        avg_elapsed = 0.0
        avg_buffer_gets = 0.0
        for s in execution_stats:
            phv = str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or '')
            if current_phv and phv == current_phv:
                avg_elapsed = _safe_float(s.get('AVG_ELAPSED_MS') or s.get('avg_elapsed_ms') or 0)
                avg_buffer_gets = _safe_float(s.get('AVG_BUFFER_GETS') or s.get('avg_buffer_gets') or 0)
                break
        if not avg_elapsed and execution_stats:
            s0 = execution_stats[0]
            avg_elapsed = _safe_float(s0.get('AVG_ELAPSED_MS') or s0.get('avg_elapsed_ms') or 0)
            avg_buffer_gets = _safe_float(s0.get('AVG_BUFFER_GETS') or s0.get('avg_buffer_gets') or 0)

        # Build assessment summary
        if plan_is_optimal:
            verdict = "CURRENT_PLAN_IS_OPTIMAL"
            explanation = (
                f"The current plan (PHV {current_phv}, {current_execs} executions) "
                f"uses index access ({', '.join(index_operations[:3])}) with no full table scans. "
                f"Avg elapsed: {avg_elapsed:.1f}ms, avg buffer gets: {avg_buffer_gets:.0f}. "
                f"No index creation or plan change is needed."
            )
        elif needs_optimization:
            verdict = "PLAN_NEEDS_OPTIMIZATION"
            explanation = (
                f"The current plan (PHV {current_phv}, {current_execs} executions) "
                f"performs full table scan on {', '.join(fts_tables)} without using any index. "
                f"Avg elapsed: {avg_elapsed:.1f}ms, avg buffer gets: {avg_buffer_gets:.0f}. "
                f"Index creation or plan change may help."
            )
        else:
            verdict = "PLAN_MIXED"
            explanation = (
                f"The current plan (PHV {current_phv}, {current_execs} executions) "
                f"uses indexes ({', '.join(index_operations[:2])}) but also has "
                f"full table scan on {', '.join(fts_tables)}. "
                f"Avg elapsed: {avg_elapsed:.1f}ms, avg buffer gets: {avg_buffer_gets:.0f}. "
                f"Review whether FTS tables are small (expected) or could benefit from index."
            )

        return {
            'verdict': verdict,
            'explanation': explanation,
            'current_plan_hash': current_phv,
            'current_plan_executions': current_execs,
            'uses_index': uses_index,
            'has_full_table_scan': has_full_table_scan,
            'index_operations': index_operations[:5],
            'fts_tables': fts_tables,
            'avg_elapsed_ms': avg_elapsed,
            'avg_buffer_gets': avg_buffer_gets,
        }

    def _analyze_postgresql_execution_plan(self, plan_text: str, sql_text: str) -> tuple[List[str], List[str]]:
        """Analyze PostgreSQL execution plan and return optimization hints and suggested SQL."""
        hints: List[str] = []
        suggestions: List[str] = []
        plan_lower = plan_text.lower()
        sql_lower = sql_text.lower()
        sql_upper = sql_text.upper()
        
        # Sequential scan detection
        seq_scans = plan_text.count('Seq Scan')
        if seq_scans > 0:
            hints.append(f"⚠️  {seq_scans} Sequential Scan(s) detected — add selective index on WHERE columns")
            suggestions.append("-- Create index on filter columns\nCREATE INDEX CONCURRENTLY idx_<table>_<col> ON <table>(<column>);")
        
        # Nested loop without index
        if 'Nested Loop' in plan_text and plan_text.count('Index') < 2:
            hints.append("⚠️  Nested Loop without index support — add index on JOIN columns")
        
        # Sort operations
        if 'Sort' in plan_text and ('ORDER BY' in sql_upper or 'GROUP BY' in sql_upper):
            hints.append("💡 Sort operation in execution plan — consider covering index for ORDER BY/GROUP BY")
            suggestions.append("-- Create covering index\nCREATE INDEX idx_<table>_<cols> ON <table>(<columns>);")
        
        # Rows removed by filter
        if 'rows removed by filter' in plan_lower:
            hints.append("❌ High filter rejection ratio — tighten WHERE predicates or add partial indexes")
        
        # Hash join analysis
        if 'Hash Join' in plan_text and seq_scans > 0:
            hints.append("ℹ️  Hash Join with sequential scans — consider indexes on join columns for index-based join")
        
        # Generate validation SQL
        suggestions.append("-- Get detailed plan with runtime stats\nEXPLAIN (ANALYZE, BUFFERS, VERBOSE) <your_query>;")
        suggestions.append("-- Update statistics if stale\nANALYZE <table>;")
        
        if not hints:
            hints.append("✅ Plan appears index-driven and efficient — monitor for performance changes")
        
        return hints, suggestions

    def _collect_oracle_index_metrics(self) -> Dict[str, Any]:
        """Collect Oracle index usage metrics."""
        unused_indexes = []
        large_indexes = []
        fragmented_indexes = []
        # Try dba_index_usage (Oracle 12.2+) first; fall back to dba_indexes without
        # the deprecated MONITORED/USED columns that were removed in 12.2.
        try:
            unused_indexes = self.conn.execute_query_dict(
                "SELECT owner, name AS index_name, total_access_count, total_rows_returned "
                "FROM dba_index_usage WHERE total_access_count = 0 FETCH FIRST 20 ROWS ONLY"
            )
        except Exception:
            try:
                unused_indexes = self.conn.execute_query_dict(
                    "SELECT owner, index_name, status, index_type "
                    "FROM dba_indexes WHERE status != 'VALID' FETCH FIRST 20 ROWS ONLY"
                )
            except Exception:
                pass
        try:
            large_indexes = self.conn.execute_query_dict(
                "SELECT owner, segment_name AS index_name, bytes "
                "FROM dba_segments WHERE segment_type='INDEX' "
                "ORDER BY bytes DESC FETCH FIRST 20 ROWS ONLY"
            )
        except Exception:
            pass

        try:
            fragmented_indexes = self.conn.execute_query_dict(
                "SELECT owner, index_name, table_name, status, blevel, leaf_blocks, num_rows, clustering_factor, "
                "ROUND(CASE WHEN NVL(num_rows, 0) = 0 THEN 0 "
                "           ELSE (clustering_factor / NULLIF(num_rows, 0)) * 100 END, 2) AS fragmentation_pct "
                "FROM dba_indexes "
                "WHERE status = 'VALID' "
                "  AND NVL(leaf_blocks, 0) > 1000 "
                "  AND NVL(num_rows, 0) > 0 "
                "ORDER BY fragmentation_pct DESC, blevel DESC "
                "FETCH FIRST 25 ROWS ONLY"
            )
        except Exception:
            fragmented_indexes = []

        return {
            'unused_indexes': unused_indexes,
            'large_indexes': large_indexes,
            'fragmented_indexes': fragmented_indexes,
            'fragmented_index_count': len(fragmented_indexes),
        }

    def _collect_oracle_table_metrics(self) -> Dict[str, Any]:
        """Collect Oracle table metrics."""
        table_stats = []
        large_tables = []
        tablespace_usage = []
        temp_tablespace_usage = []
        try:
            table_stats = self.conn.execute_query_dict(
                "SELECT owner, table_name, num_rows FROM dba_tables WHERE num_rows IS NOT NULL ORDER BY num_rows DESC FETCH FIRST 20 ROWS ONLY"
            )
            large_tables = self.conn.execute_query_dict(
                "SELECT owner, segment_name AS tablename, bytes FROM dba_segments "
                "WHERE segment_type='TABLE' ORDER BY bytes DESC FETCH FIRST 20 ROWS ONLY"
            )
        except Exception:
            pass

        try:
            tablespace_usage = self.conn.execute_query_dict(
                "SELECT df.tablespace_name, "
                "       ROUND(df.total_mb, 2) AS total_mb, "
                "       ROUND(NVL(fs.free_mb, 0), 2) AS free_mb, "
                "       ROUND(df.total_mb - NVL(fs.free_mb, 0), 2) AS used_mb, "
                "       ROUND(CASE WHEN df.total_mb = 0 THEN 0 "
                "                  ELSE ((df.total_mb - NVL(fs.free_mb, 0)) / df.total_mb) * 100 END, 2) AS used_pct "
                "FROM (SELECT tablespace_name, SUM(bytes)/1024/1024 AS total_mb "
                "      FROM dba_data_files GROUP BY tablespace_name) df "
                "LEFT JOIN (SELECT tablespace_name, SUM(bytes)/1024/1024 AS free_mb "
                "           FROM dba_free_space GROUP BY tablespace_name) fs "
                "  ON df.tablespace_name = fs.tablespace_name "
                "ORDER BY used_pct DESC"
            )
        except Exception:
            tablespace_usage = []

        try:
            temp_tablespace_usage = self.conn.execute_query_dict(
                "SELECT tf.tablespace_name, "
                "       ROUND(SUM(tf.bytes)/1024/1024, 2) AS total_mb, "
                "       ROUND(SUM(NVL(th.bytes_free, 0))/1024/1024, 2) AS free_mb, "
                "       ROUND((SUM(tf.bytes) - SUM(NVL(th.bytes_free, 0)))/1024/1024, 2) AS used_mb, "
                "       ROUND(CASE WHEN SUM(tf.bytes) = 0 THEN 0 "
                "                  ELSE ((SUM(tf.bytes) - SUM(NVL(th.bytes_free, 0))) / SUM(tf.bytes)) * 100 END, 2) AS used_pct "
                "FROM dba_temp_files tf "
                "LEFT JOIN v$temp_space_header th "
                "  ON tf.tablespace_name = th.tablespace_name "
                "GROUP BY tf.tablespace_name "
                "ORDER BY used_pct DESC"
            )
        except Exception:
            temp_tablespace_usage = []

        return {
            'table_stats': table_stats,
            'large_tables': large_tables,
            'tablespace_usage': tablespace_usage,
            'temp_tablespace_usage': temp_tablespace_usage,
        }

    def _collect_oracle_lock_metrics(self) -> Dict[str, Any]:
        """Collect Oracle lock contention metrics."""
        waiting_locks = []
        lock_statistics = []
        try:
            waiting_locks = self.conn.execute_query_dict(
                "SELECT sid, serial#, event, wait_class FROM v$session WHERE state='ACTIVE' AND wait_class != 'Idle'"
            )
            lock_statistics = self.conn.execute_query_dict(
                "SELECT s.sid AS session_id, s.username AS oracle_username, "
                "l.type AS lock_type, l.lmode AS locked_mode, l.request, l.id1, l.id2 "
                "FROM v$lock l JOIN v$session s ON l.sid = s.sid "
                "WHERE l.lmode > 0 AND s.username IS NOT NULL "
                "FETCH FIRST 20 ROWS ONLY"
            )
        except Exception:
            pass

        return {
            'waiting_locks': waiting_locks,
            'lock_statistics': lock_statistics
        }

    def _collect_oracle_replication_metrics(self) -> Dict[str, Any]:
        """Collect Oracle replication metrics (if available)."""
        replication = []
        try:
            # v$dataguard_stats is available on both primary and standby; safe on non-DG setups
            replication = self.conn.execute_query_dict(
                "SELECT name, value, unit, time_computed "
                "FROM v$dataguard_stats WHERE rownum <= 20"
            )
        except Exception:
            pass

        return {
            'replication_slots': replication
        }

    # ──────────────────────────────────────────────────────────────────────
    # Troubleshooting: Wait Events, Blocking Tree, ASH
    # ──────────────────────────────────────────────────────────────────────

    def collect_wait_event_analysis(self) -> Dict[str, Any]:
        """Collect wait event analysis for Oracle or PostgreSQL."""
        db_type = self.conn.get_database_type().lower()
        if db_type == 'oracle':
            return self._collect_oracle_wait_events()
        return self._collect_pg_wait_events()

    def _collect_oracle_wait_events(self) -> Dict[str, Any]:
        """Collect Oracle wait event analysis from V$ views.

        Optimized: batches all 4 queries in a single JVM session when the
        MCP connector is available, cutting wall-clock time by ~90 %.
        """
        batch_queries = {
            "system_waits": (
                "SELECT event, wait_class, total_waits, total_timeouts, "
                "time_waited, average_wait, time_waited_micro "
                "FROM v$system_event "
                "WHERE wait_class NOT IN ('Idle') "
                "ORDER BY time_waited DESC FETCH FIRST 20 ROWS ONLY"
            ),
            "session_waits": (
                "SELECT s.sid, s.serial#, s.username, s.machine, s.program, "
                "s.event, s.wait_class, s.state, "
                "ROUND(s.seconds_in_wait) AS seconds_waiting, "
                "s.sql_id, SUBSTR(sq.sql_text, 1, 200) AS sql_text "
                "FROM v$session s "
                "LEFT JOIN v$sql sq ON sq.sql_id = s.sql_id AND sq.child_number = s.sql_child_number "
                "WHERE s.wait_class NOT IN ('Idle') AND s.state = 'WAITING' "
                "ORDER BY s.seconds_in_wait DESC FETCH FIRST 25 ROWS ONLY"
            ),
            "wait_class_summary": (
                "SELECT wait_class, "
                "SUM(total_waits) AS total_waits, "
                "SUM(time_waited) AS total_time_waited, "
                "ROUND(SUM(time_waited) / NULLIF(SUM(total_waits), 0), 2) AS avg_wait "
                "FROM v$system_event "
                "WHERE wait_class NOT IN ('Idle') "
                "GROUP BY wait_class ORDER BY total_time_waited DESC"
            ),
            "wait_histogram": (
                "SELECT event, wait_class, wait_count, wait_time_milli "
                "FROM v$event_histogram "
                "WHERE wait_class NOT IN ('Idle') "
                "ORDER BY wait_time_milli DESC FETCH FIRST 30 ROWS ONLY"
            ),
        }

        if hasattr(self.conn, 'execute_batch_queries_dict'):
            _t0 = time.monotonic()
            batch = self.conn.execute_batch_queries_dict(batch_queries)
            logger.info("Wait-events batched: 4 queries in 1 JVM, %.0f ms",
                        (time.monotonic() - _t0) * 1000)
            return {
                'database_type': 'oracle',
                'system_waits': batch.get('system_waits', []),
                'session_waits': batch.get('session_waits', []),
                'wait_class_summary': batch.get('wait_class_summary', []),
                'wait_histogram': batch.get('wait_histogram', []),
            }

        # Fallback: individual queries (non-MCP connections)
        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("Wait event query failed: %s", ex)
                return []

        return {
            'database_type': 'oracle',
            'system_waits': _q(batch_queries["system_waits"]),
            'session_waits': _q(batch_queries["session_waits"]),
            'wait_class_summary': _q(batch_queries["wait_class_summary"]),
            'wait_histogram': _q(batch_queries["wait_histogram"]),
        }

    def _collect_pg_wait_events(self) -> Dict[str, Any]:
        """Collect PostgreSQL wait event analysis."""
        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("PG wait event query failed: %s", ex)
                return []

        # Active wait events from pg_stat_activity
        session_waits = _q(
            "SELECT pid, usename, application_name, client_addr, "
            "wait_event_type, wait_event, state, "
            "EXTRACT(EPOCH FROM (now() - query_start))::INT AS seconds_waiting, "
            "LEFT(query, 200) AS query_text "
            "FROM pg_stat_activity "
            "WHERE wait_event IS NOT NULL AND state != 'idle' "
            "ORDER BY query_start ASC LIMIT 25"
        )

        # Wait event type summary
        wait_class_summary = _q(
            "SELECT wait_event_type AS wait_class, "
            "wait_event, COUNT(*) AS session_count "
            "FROM pg_stat_activity "
            "WHERE wait_event IS NOT NULL AND state != 'idle' "
            "GROUP BY wait_event_type, wait_event "
            "ORDER BY session_count DESC LIMIT 20"
        )

        return {
            'database_type': 'postgresql',
            'session_waits': session_waits,
            'wait_class_summary': wait_class_summary,
            'system_waits': [],
            'wait_histogram': [],
        }

    def collect_blocking_tree(self) -> Dict[str, Any]:
        """Collect blocking session tree for Oracle or PostgreSQL."""
        db_type = self.conn.get_database_type().lower()
        if db_type == 'oracle':
            return self._collect_oracle_blocking_tree()
        return self._collect_pg_blocking_tree()

    def _collect_oracle_blocking_tree(self) -> Dict[str, Any]:
        """Build Oracle blocking session tree from V$SESSION.

        Optimized: batches all 3 queries in a single JVM session.
        """
        batch_queries = {
            "blocking_chain": (
                "SELECT "
                "  LEVEL AS tree_level, "
                "  s.sid, s.serial#, s.username, s.machine, s.program, "
                "  s.status, s.sql_id, "
                "  s.event AS wait_event, s.wait_class, "
                "  ROUND(s.seconds_in_wait) AS seconds_waiting, "
                "  s.blocking_session AS blocked_by_sid, "
                "  SUBSTR(sq.sql_text, 1, 200) AS sql_text "
                "FROM v$session s "
                "LEFT JOIN v$sql sq ON sq.sql_id = s.sql_id AND sq.child_number = s.sql_child_number "
                "START WITH s.blocking_session IS NOT NULL "
                "  AND NOT EXISTS (SELECT 1 FROM v$session s2 WHERE s2.sid = s.blocking_session AND s2.blocking_session IS NOT NULL) "
                "CONNECT BY PRIOR s.sid = s.blocking_session "
                "ORDER SIBLINGS BY s.seconds_in_wait DESC"
            ),
            "root_blockers": (
                "SELECT DISTINCT b.sid, b.serial#, b.username, b.machine, "
                "b.program, b.status, b.sql_id, b.event AS wait_event, "
                "SUBSTR(sq.sql_text, 1, 200) AS sql_text, "
                "(SELECT COUNT(*) FROM v$session w WHERE w.blocking_session = b.sid) AS blocked_count "
                "FROM v$session b "
                "JOIN v$session w ON w.blocking_session = b.sid "
                "LEFT JOIN v$sql sq ON sq.sql_id = b.sql_id AND sq.child_number = b.sql_child_number "
                "WHERE b.blocking_session IS NULL "
                "ORDER BY blocked_count DESC"
            ),
            "waiters": (
                "SELECT w.sid AS waiting_sid, w.serial# AS waiting_serial, "
                "w.username AS waiting_user, w.machine AS waiting_machine, "
                "w.event AS wait_event, "
                "ROUND(w.seconds_in_wait) AS seconds_waiting, "
                "w.blocking_session AS blocked_by_sid, "
                "w.sql_id, SUBSTR(sq.sql_text, 1, 200) AS sql_text "
                "FROM v$session w "
                "LEFT JOIN v$sql sq ON sq.sql_id = w.sql_id AND sq.child_number = w.sql_child_number "
                "WHERE w.blocking_session IS NOT NULL "
                "ORDER BY w.seconds_in_wait DESC"
            ),
        }

        if hasattr(self.conn, 'execute_batch_queries_dict'):
            _t0 = time.monotonic()
            batch = self.conn.execute_batch_queries_dict(batch_queries)
            logger.info("Blocking-tree batched: 3 queries in 1 JVM, %.0f ms",
                        (time.monotonic() - _t0) * 1000)
            return {
                'database_type': 'oracle',
                'blocking_chain': batch.get('blocking_chain', []),
                'root_blockers': batch.get('root_blockers', []),
                'waiters': batch.get('waiters', []),
            }

        # Fallback: individual queries
        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("Blocking tree query failed: %s", ex)
                return []

        return {
            'database_type': 'oracle',
            'blocking_chain': _q(batch_queries["blocking_chain"]),
            'root_blockers': _q(batch_queries["root_blockers"]),
            'waiters': _q(batch_queries["waiters"]),
        }

    def _collect_pg_blocking_tree(self) -> Dict[str, Any]:
        """Build PostgreSQL blocking session tree."""
        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("PG blocking tree query failed: %s", ex)
                return []

        blocking_chain = _q(
            "SELECT "
            "  blocked.pid AS waiting_pid, "
            "  blocked.usename AS waiting_user, "
            "  blocked.application_name AS waiting_app, "
            "  LEFT(blocked.query, 200) AS waiting_query, "
            "  blocked.wait_event_type, blocked.wait_event, "
            "  EXTRACT(EPOCH FROM (now() - blocked.query_start))::INT AS seconds_waiting, "
            "  blocking.pid AS blocking_pid, "
            "  blocking.usename AS blocking_user, "
            "  blocking.application_name AS blocking_app, "
            "  LEFT(blocking.query, 200) AS blocking_query, "
            "  blocking.state AS blocking_state "
            "FROM pg_stat_activity blocked "
            "JOIN LATERAL unnest(pg_blocking_pids(blocked.pid)) AS bp(pid) ON TRUE "
            "JOIN pg_stat_activity blocking ON blocking.pid = bp.pid "
            "WHERE blocked.pid != blocked.backend_xid::text::int OR TRUE "
            "ORDER BY seconds_waiting DESC"
        )

        return {
            'database_type': 'postgresql',
            'blocking_chain': blocking_chain,
            'root_blockers': [],
            'waiters': [],
        }

    def collect_ash_data(self, minutes_back: int = 30) -> Dict[str, Any]:
        """Collect ASH (Active Session History) data for Oracle or PostgreSQL."""
        db_type = self.conn.get_database_type().lower()
        if db_type == 'oracle':
            return self._collect_oracle_ash(minutes_back)
        return self._collect_pg_ash(minutes_back)

    def _collect_oracle_ash(self, minutes_back: int = 30) -> Dict[str, Any]:
        """Collect Oracle ASH data from V$ACTIVE_SESSION_HISTORY.

        Optimized: batches all queries (top SQL, events, sessions, timeline,
        SQL text enrichment) into a single JVM session.
        """
        safe_minutes = max(1, min(minutes_back, 1440))

        batch_queries: Dict[str, str] = {
            "top_sql": (
                f"SELECT sql_id, COUNT(*) AS sample_count, "
                f"ROUND(COUNT(*) * 100 / NULLIF(SUM(COUNT(*)) OVER (), 0), 1) AS pct_db_time, "
                f"MIN(session_id) AS example_sid, "
                f"MAX(event) AS last_event, MAX(wait_class) AS last_wait_class "
                f"FROM v$active_session_history "
                f"WHERE sample_time > SYSDATE - {safe_minutes}/1440 "
                f"AND sql_id IS NOT NULL "
                f"GROUP BY sql_id ORDER BY sample_count DESC "
                f"FETCH FIRST 15 ROWS ONLY"
            ),
            "top_events": (
                f"SELECT NVL(event, 'On CPU') AS event, wait_class, "
                f"COUNT(*) AS sample_count, "
                f"ROUND(COUNT(*) * 100 / NULLIF(SUM(COUNT(*)) OVER (), 0), 1) AS pct_db_time "
                f"FROM v$active_session_history "
                f"WHERE sample_time > SYSDATE - {safe_minutes}/1440 "
                f"GROUP BY event, wait_class ORDER BY sample_count DESC "
                f"FETCH FIRST 15 ROWS ONLY"
            ),
            "top_sessions": (
                f"SELECT session_id AS sid, session_serial# AS serial, "
                f"NVL(user_id, 0) AS user_id, "
                f"module, action, program, "
                f"COUNT(*) AS sample_count, "
                f"ROUND(COUNT(*) * 100 / NULLIF(SUM(COUNT(*)) OVER (), 0), 1) AS pct_db_time "
                f"FROM v$active_session_history "
                f"WHERE sample_time > SYSDATE - {safe_minutes}/1440 "
                f"GROUP BY session_id, session_serial#, user_id, module, action, program "
                f"ORDER BY sample_count DESC FETCH FIRST 15 ROWS ONLY"
            ),
            "activity_timeline": (
                f"SELECT TO_CHAR(TRUNC(sample_time, 'MI'), 'HH24:MI') AS time_bucket, "
                f"NVL(wait_class, 'CPU') AS wait_class, "
                f"COUNT(*) AS sample_count "
                f"FROM v$active_session_history "
                f"WHERE sample_time > SYSDATE - {safe_minutes}/1440 "
                f"GROUP BY TRUNC(sample_time, 'MI'), wait_class "
                f"ORDER BY TRUNC(sample_time, 'MI')"
            ),
            # Pre-fetch SQL text for enrichment (top 10 sql_ids inline via subquery)
            "sql_text": (
                f"SELECT sql_id, SUBSTR(sql_text, 1, 300) AS sql_text "
                f"FROM v$sql "
                f"WHERE sql_id IN ("
                f"  SELECT sql_id FROM ("
                f"    SELECT sql_id, COUNT(*) AS cnt "
                f"    FROM v$active_session_history "
                f"    WHERE sample_time > SYSDATE - {safe_minutes}/1440 "
                f"      AND sql_id IS NOT NULL "
                f"    GROUP BY sql_id ORDER BY cnt DESC "
                f"    FETCH FIRST 10 ROWS ONLY"
                f"  )"
                f") AND rownum <= 10"
            ),
        }

        if hasattr(self.conn, 'execute_batch_queries_dict'):
            _t0 = time.monotonic()
            batch = self.conn.execute_batch_queries_dict(batch_queries)
            logger.info("ASH data batched: %d queries in 1 JVM, %.0f ms",
                        len(batch_queries), (time.monotonic() - _t0) * 1000)

            top_sql_text = {}
            for r in batch.get('sql_text', []):
                sid = str(r.get('SQL_ID') or r.get('sql_id') or '')
                txt = str(r.get('SQL_TEXT') or r.get('sql_text') or '')
                if sid:
                    top_sql_text[sid] = txt

            return {
                'database_type': 'oracle',
                'minutes_back': safe_minutes,
                'top_sql': batch.get('top_sql', []),
                'top_events': batch.get('top_events', []),
                'top_sessions': batch.get('top_sessions', []),
                'activity_timeline': batch.get('activity_timeline', []),
                'sql_text_map': top_sql_text,
            }

        # Fallback: individual queries (non-MCP connections)
        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("ASH query failed: %s", ex)
                return []

        top_sql = _q(batch_queries["top_sql"])
        top_sql_text = {}
        sql_ids = [str(r.get('SQL_ID') or r.get('sql_id') or '') for r in top_sql[:10] if r.get('SQL_ID') or r.get('sql_id')]
        if sql_ids:
            id_list = ",".join(f"'{sid}'" for sid in sql_ids if sid)
            sql_texts = _q(
                f"SELECT sql_id, SUBSTR(sql_text, 1, 300) AS sql_text "
                f"FROM v$sql WHERE sql_id IN ({id_list}) "
                f"AND rownum <= {len(sql_ids)}"
            )
            for r in sql_texts:
                sid = str(r.get('SQL_ID') or r.get('sql_id') or '')
                txt = str(r.get('SQL_TEXT') or r.get('sql_text') or '')
                if sid:
                    top_sql_text[sid] = txt

        return {
            'database_type': 'oracle',
            'minutes_back': safe_minutes,
            'top_sql': top_sql,
            'top_events': _q(batch_queries["top_events"]),
            'top_sessions': _q(batch_queries["top_sessions"]),
            'activity_timeline': _q(batch_queries["activity_timeline"]),
            'sql_text_map': top_sql_text,
        }

    def _collect_pg_ash(self, minutes_back: int = 30) -> Dict[str, Any]:
        """Collect PostgreSQL ASH-equivalent from pg_stat_activity snapshots."""
        safe_minutes = max(1, min(minutes_back, 1440))

        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("PG ASH query failed: %s", ex)
                return []

        # Current active sessions (PostgreSQL has no persistent ASH — snapshot current state)
        active_sessions = _q(
            "SELECT pid, usename, application_name, client_addr, "
            "wait_event_type, wait_event, state, "
            "LEFT(query, 300) AS query_text, "
            "EXTRACT(EPOCH FROM (now() - query_start))::INT AS elapsed_seconds, "
            "EXTRACT(EPOCH FROM (now() - xact_start))::INT AS xact_seconds "
            "FROM pg_stat_activity "
            "WHERE state NOT IN ('idle') AND pid != pg_backend_pid() "
            "ORDER BY query_start ASC LIMIT 30"
        )

        # pg_stat_statements top SQL (historical proxy for ASH)
        top_sql = _q(
            "SELECT queryid, LEFT(query, 300) AS query_text, "
            "calls, total_exec_time, mean_exec_time, rows "
            "FROM pg_stat_statements "
            "WHERE query NOT LIKE '%pg_stat%' "
            "ORDER BY total_exec_time DESC LIMIT 15"
        )

        # Wait event summary
        wait_summary = _q(
            "SELECT wait_event_type, wait_event, COUNT(*) AS session_count "
            "FROM pg_stat_activity "
            "WHERE wait_event IS NOT NULL AND state != 'idle' "
            "GROUP BY wait_event_type, wait_event "
            "ORDER BY session_count DESC LIMIT 15"
        )

        return {
            'database_type': 'postgresql',
            'minutes_back': safe_minutes,
            'top_sql': top_sql,
            'top_events': wait_summary,
            'top_sessions': active_sessions,
            'activity_timeline': [],
            'sql_text_map': {},
        }

    # ──────────────────────────────────────────────────────────────────────
    # SQL ID / Session ID deep-dive lookup
    # ──────────────────────────────────────────────────────────────────────

    def collect_sqlid_info(self, sql_id: str) -> Dict[str, Any]:
        """Collect comprehensive information about a specific SQL ID."""
        db_type = self.conn.get_database_type().lower()
        if db_type == 'oracle':
            return self._collect_oracle_sqlid_info(sql_id)
        return self._collect_pg_sqlid_info(sql_id)

    def _collect_oracle_sqlid_info(self, sql_id: str) -> Dict[str, Any]:
        """Collect Oracle SQL ID details from V$ views.

        Optimized: batches all non-dependent queries into a single JVM session,
        then runs a conditional second batch only for plan fallback layers.
        """
        raw_id = str(sql_id).strip()
        safe_id = ''.join(ch for ch in raw_id if ch.isalnum())[:30]

        if not safe_id:
            return {
                'database_type': 'oracle',
                'sql_id': '',
                'sql_text': [],
                'execution_stats': [],
                'execution_history_daily': [],
                'execution_plan': [],
                'ash_activity': [],
                'active_sessions': [],
                'session_wait_details': [],
                'session_statistics': [],
                'transaction_rollback_info': [],
                'sort_usage_info': [],
                'sql_monitor': [],
                'bind_variables': [],
                'available_plan_hash_values': [],
            }

        # ── Build all independent queries for a single batch ────────────
        batch_queries: Dict[str, str] = {
            "sql_text": (
                f"SELECT sql_id, sql_fulltext, parsing_schema_name, "
                f"first_load_time, last_load_time, last_active_time "
                f"FROM v$sql WHERE sql_id = '{safe_id}' AND ROWNUM = 1"
            ),
            "exec_stats_awr": (
                f"SELECT hs.sql_id, "
                f"       0 AS child_number, "
                f"       hs.plan_hash_value, "
                f"       SUM(hs.executions_delta) AS executions, "
                f"       SUM(hs.elapsed_time_delta) AS elapsed_time, "
                f"       SUM(hs.cpu_time_delta) AS cpu_time, "
                f"       SUM(hs.buffer_gets_delta) AS buffer_gets, "
                f"       SUM(hs.disk_reads_delta) AS disk_reads, "
                f"       SUM(hs.rows_processed_delta) AS rows_processed, "
                f"       0 AS sorts, "
                f"       ROUND(SUM(hs.elapsed_time_delta)/NULLIF(SUM(hs.executions_delta),0)/1000, 2) AS avg_elapsed_ms, "
                f"       ROUND(SUM(hs.cpu_time_delta)/NULLIF(SUM(hs.executions_delta),0)/1000, 2) AS avg_cpu_ms, "
                f"       ROUND(SUM(hs.buffer_gets_delta)/NULLIF(SUM(hs.executions_delta),0), 1) AS avg_buffer_gets, "
                f"       ROUND(SUM(hs.disk_reads_delta)/NULLIF(SUM(hs.executions_delta),0), 1) AS avg_disk_reads, "
                f"       ROUND(SUM(hs.rows_processed_delta)/NULLIF(SUM(hs.executions_delta),0), 1) AS avg_rows, "
                f"       TO_CHAR(MAX(sn.end_interval_time), 'YYYY-MM-DD HH24:MI:SS') AS last_active_time "
                f"FROM awr_root_sqlstat hs "
                f"JOIN awr_root_snapshot sn "
                f"  ON sn.dbid = hs.dbid "
                f" AND sn.instance_number = hs.instance_number "
                f" AND sn.snap_id = hs.snap_id "
                f"WHERE hs.sql_id = '{safe_id}' "
                f"  AND hs.plan_hash_value IS NOT NULL AND hs.plan_hash_value != 0 "
                f"  AND sn.begin_interval_time >= SYSDATE - 30 "
                f"GROUP BY hs.sql_id, hs.plan_hash_value "
                f"ORDER BY SUM(hs.elapsed_time_delta) DESC"
            ),
            "execution_history_daily": (
                f"SELECT * "
                f"FROM ( "
                f"  SELECT TO_CHAR(TRUNC(s.begin_interval_time), 'YYYY-MM-DD') AS exec_date, "
                f"         t.plan_hash_value, "
                f"         t.instance_number AS inst_id, "
                f"         t.module, "
                f"         SUM(t.executions_delta) AS execs, "
                f"         SUM(t.buffer_gets_delta) AS buffer_gets, "
                f"         SUM(t.rows_processed_delta) AS rows_processed, "
                f"         ROUND(SUM(t.cpu_time_delta) / 1000000, 0) AS cpu_tim_secs, "
                f"         ROUND(SUM(t.elapsed_time_delta) / 1000000, 0) AS ela_tim_secs, "
                f"         ROUND(SUM(t.elapsed_time_delta) "
                f"               / DECODE(SUM(t.executions_delta), 0, 1, SUM(t.executions_delta)) "
                f"               / 1000000, 0) AS avg_ela_secs, "
                f"         SUM(t.px_servers_execs_delta) AS px_tot "
                f"  FROM awr_root_sqlstat t "
                f"  JOIN awr_root_snapshot s "
                f"    ON t.snap_id = s.snap_id "
                f"   AND t.instance_number = s.instance_number "
                f"  WHERE t.sql_id = '{safe_id}' "
                f"  GROUP BY TRUNC(s.begin_interval_time), "
                f"           t.plan_hash_value, "
                f"           t.instance_number, "
                f"           t.module "
                f"  ORDER BY TRUNC(s.begin_interval_time) DESC "
                f") "
                f"FETCH FIRST 40 ROWS ONLY"
            ),
            "exec_stats_cursor": (
                f"SELECT sql_id, child_number, plan_hash_value, "
                f"executions, elapsed_time, cpu_time, buffer_gets, "
                f"disk_reads, rows_processed, sorts, "
                f"ROUND(elapsed_time/NULLIF(executions,0)/1000, 2) AS avg_elapsed_ms, "
                f"ROUND(cpu_time/NULLIF(executions,0)/1000, 2) AS avg_cpu_ms, "
                f"ROUND(buffer_gets/NULLIF(executions,0), 1) AS avg_buffer_gets, "
                f"ROUND(disk_reads/NULLIF(executions,0), 1) AS avg_disk_reads, "
                f"ROUND(rows_processed/NULLIF(executions,0), 1) AS avg_rows, "
                f"last_active_time, first_load_time "
                f"FROM v$sql WHERE sql_id = '{safe_id}' "
                f"ORDER BY child_number"
            ),
            "plan_xplan": (
                f"SELECT PLAN_TABLE_OUTPUT FROM TABLE("
                f"DBMS_XPLAN.DISPLAY_CURSOR('{safe_id}', NULL, 'ALLSTATS LAST'))"
            ),
            "plan_vsqlplan": (
                f"SELECT id, parent_id, operation, options, object_owner, object_name, "
                f"object_type, cost, cardinality, bytes, cpu_cost, io_cost, "
                f"temp_space, access_predicates, filter_predicates, depth "
                f"FROM v$sql_plan WHERE sql_id = '{safe_id}' "
                f"AND child_number = (SELECT MIN(child_number) FROM v$sql_plan WHERE sql_id = '{safe_id}') "
                f"ORDER BY id"
            ),
            "all_plan_hashes": (
                f"SELECT DISTINCT child_number, plan_hash_value "
                f"FROM v$sql_plan WHERE sql_id = '{safe_id}' "
                f"ORDER BY child_number"
            ),
            "all_plans_vsqlplan": (
                f"SELECT child_number, plan_hash_value, "
                f"id, parent_id, operation, options, object_owner, object_name, "
                f"object_type, cost, cardinality, bytes, cpu_cost, io_cost, "
                f"temp_space, access_predicates, filter_predicates, depth "
                f"FROM v$sql_plan WHERE sql_id = '{safe_id}' "
                f"ORDER BY child_number, id"
            ),
            "ash_activity": (
                f"SELECT NVL(event, 'On CPU') AS event, wait_class, "
                f"COUNT(*) AS sample_count, "
                f"MIN(sample_time) AS first_seen, MAX(sample_time) AS last_seen "
                f"FROM v$active_session_history "
                f"WHERE sql_id = '{safe_id}' AND sample_time > SYSDATE - 1/24 "
                f"GROUP BY event, wait_class ORDER BY sample_count DESC"
            ),
            "active_sessions": (
                f"SELECT s.sid, s.serial#, s.username, s.osuser, s.machine, s.process, "
                f"s.program, s.module, s.action, "
                f"p.pid AS oracle_pid, p.spid, "
                f"s.status, s.event, s.wait_class, s.state, s.blocking_session, "
                f"ROUND(s.seconds_in_wait) AS seconds_waiting, "
                f"TO_CHAR(s.logon_time, 'MM/DD HH24:MI') AS logon_time, "
                f"s.sql_exec_start "
                f"FROM v$session s "
                f"LEFT JOIN v$process p ON p.addr = s.paddr "
                f"WHERE s.sql_id = '{safe_id}'"
            ),
            "session_wait_details": (
                f"SELECT s.sid, s.serial#, "
                f"       w.event, w.p1text, w.p1, w.p2text, w.p2, w.p3text, w.p3 "
                f"FROM v$session s "
                f"JOIN v$session_wait w ON w.sid = s.sid "
                f"WHERE s.sql_id = '{safe_id}' "
                f"  AND w.wait_time = 0 "
                f"ORDER BY s.sid"
            ),
            "session_statistics": (
                f"SELECT s.sid, s.serial#, b.name, "
                f"       CASE WHEN b.name = 'redo size' THEN ROUND(a.value/1024/1024, 2) ELSE a.value END AS value, "
                f"       CASE WHEN b.name = 'redo size' THEN 'MB' ELSE NULL END AS unit "
                f"FROM v$session s "
                f"JOIN v$sesstat a ON a.sid = s.sid "
                f"JOIN v$statname b ON b.statistic# = a.statistic# "
                f"WHERE s.sql_id = '{safe_id}' "
                f"  AND b.name IN ('redo size', 'parse count (total)', 'parse count (hard)', 'user commits') "
                f"ORDER BY s.sid, DECODE(b.name, 'redo size', 1, 2), b.name"
            ),
            "transaction_rollback": (
                f"SELECT s.sid, s.serial#, t.used_ublk, t.used_urec, t.xidusn, "
                f"       r.name AS rollback_segment_name, t.log_io, t.phy_io, "
                f"       t.start_uext, t.start_time, t.status "
                f"FROM v$session s "
                f"JOIN v$transaction t ON t.addr = s.taddr "
                f"LEFT JOIN v$rollname r ON r.usn = t.xidusn "
                f"WHERE s.sql_id = '{safe_id}'"
            ),
            "sort_usage": (
                f"SELECT s.sid, s.serial#, u.tablespace, u.contents, u.extents, u.blocks, "
                f"       ROUND((u.blocks * 8) / 1024, 2) AS sort_space_mb "
                f"FROM v$session s "
                f"JOIN v$sort_usage u ON u.session_addr = s.saddr "
                f"WHERE s.sql_id = '{safe_id}'"
            ),
            "sql_monitor": (
                f"SELECT sql_id, status, elapsed_time, cpu_time, "
                f"buffer_gets, disk_reads, sql_exec_start "
                f"FROM v$sql_monitor WHERE sql_id = '{safe_id}' "
                f"AND ROWNUM <= 5 ORDER BY sql_exec_start DESC"
            ),

            "signature": (
                f"SELECT force_matching_signature, plan_hash_value, "
                f"       ROUND(elapsed_time / NULLIF(executions, 0) / 1e6, 4) AS avg_elapsed_sec "
                f"FROM v$sql WHERE sql_id = '{safe_id}' AND rownum <= 1"
            ),
            "awr_plan_hashes": (
                f"SELECT DISTINCT plan_hash_value "
                f"FROM awr_root_sql_plan WHERE sql_id = '{safe_id}' "
                f"AND plan_hash_value != 0"
            ),
        }

        # ── Execute in one JVM session ──────────────────────────────────
        if hasattr(self.conn, 'execute_batch_queries_dict'):
            _t0 = time.monotonic()
            batch = self.conn.execute_batch_queries_dict(batch_queries)
            logger.info("SQL-ID info batched: %d queries in 1 JVM, %.0f ms",
                        len(batch_queries), (time.monotonic() - _t0) * 1000)
        else:
            # Fallback: individual queries
            batch = {}
            for label, sql_text in batch_queries.items():
                try:
                    batch[label] = self.conn.execute_query_dict(sql_text) or []
                except Exception as ex:
                    logger.warning("SQL ID query failed [%s]: %s", label, ex)
                    batch[label] = []

        # ── Helper to filter invalid plan_hash_value rows ──
        def _valid_phv(row):
            phv = row.get('PLAN_HASH_VALUE') or row.get('plan_hash_value')
            return phv is not None and str(phv).strip() not in ('', '0', 'None')

        # ── AWR view fallback: ALWAYS try awr_root_* to catch missing plan hashes ──
        # awr_root_sqlstat may not have all plan hashes (e.g. due to date filter or CDB/PDB).
        # awr_root_sqlstat has no date filter and covers root-level AWR data.
        awr_retry: Dict[str, str] = {}
        awr_retry["awr_root_exec_stats"] = (
            f"SELECT hs.sql_id, "
            f"       0 AS child_number, "
            f"       hs.plan_hash_value, "
            f"       SUM(hs.executions_delta) AS executions, "
            f"       SUM(hs.elapsed_time_delta) AS elapsed_time, "
            f"       SUM(hs.cpu_time_delta) AS cpu_time, "
            f"       SUM(hs.buffer_gets_delta) AS buffer_gets, "
            f"       SUM(hs.disk_reads_delta) AS disk_reads, "
            f"       SUM(hs.rows_processed_delta) AS rows_processed, "
            f"       0 AS sorts, "
            f"       ROUND(SUM(hs.elapsed_time_delta)/NULLIF(SUM(hs.executions_delta),0)/1000, 2) AS avg_elapsed_ms, "
            f"       ROUND(SUM(hs.cpu_time_delta)/NULLIF(SUM(hs.executions_delta),0)/1000, 2) AS avg_cpu_ms, "
            f"       ROUND(SUM(hs.buffer_gets_delta)/NULLIF(SUM(hs.executions_delta),0), 1) AS avg_buffer_gets, "
            f"       ROUND(SUM(hs.disk_reads_delta)/NULLIF(SUM(hs.executions_delta),0), 1) AS avg_disk_reads, "
            f"       ROUND(SUM(hs.rows_processed_delta)/NULLIF(SUM(hs.executions_delta),0), 1) AS avg_rows, "
            f"       TO_CHAR(MAX(sn.end_interval_time), 'YYYY-MM-DD HH24:MI:SS') AS last_active_time "
            f"FROM awr_root_sqlstat hs "
            f"JOIN awr_root_snapshot sn "
            f"  ON sn.dbid = hs.dbid "
            f" AND sn.instance_number = hs.instance_number "
            f" AND sn.snap_id = hs.snap_id "
            f"WHERE hs.sql_id = '{safe_id}' "
            f"  AND hs.plan_hash_value IS NOT NULL AND hs.plan_hash_value != 0 "
            f"GROUP BY hs.sql_id, hs.plan_hash_value "
            f"ORDER BY SUM(hs.elapsed_time_delta) DESC"
        )
        if not batch.get('awr_plan_hashes'):
            awr_retry["awr_plan_hashes"] = (
                f"SELECT DISTINCT plan_hash_value "
                f"FROM awr_root_sql_plan WHERE sql_id = '{safe_id}' "
                f"AND plan_hash_value != 0"
            )
        # AWR sql_text fallback when cursor cache has no sql_text
        if not batch.get('sql_text'):
            awr_retry["sql_text_awr"] = (
                f"SELECT sql_id, sql_text AS sql_fulltext, "
                f"'' AS parsing_schema_name, '' AS first_load_time, "
                f"'' AS last_load_time, '' AS last_active_time "
                f"FROM awr_root_sqltext WHERE sql_id = '{safe_id}' AND ROWNUM = 1"
            )
            awr_retry["sql_text_awr_root"] = (
                f"SELECT sql_id, sql_text AS sql_fulltext, "
                f"'' AS parsing_schema_name, '' AS first_load_time, "
                f"'' AS last_load_time, '' AS last_active_time "
                f"FROM awr_root_sqltext WHERE sql_id = '{safe_id}' AND ROWNUM = 1"
            )
        if hasattr(self.conn, 'execute_batch_queries_dict'):
            awr_fb2 = self.conn.execute_batch_queries_dict(awr_retry)
        else:
            awr_fb2 = {}
            for lbl, sql in awr_retry.items():
                try:
                    awr_fb2[lbl] = self.conn.execute_query_dict(sql) or []
                except Exception:
                    awr_fb2[lbl] = []
        # Merge awr_root exec stats into exec_stats_awr (append rows for phvs not yet seen)
        awr_root_stats = [s for s in awr_fb2.get('awr_root_exec_stats', []) if _valid_phv(s)]
        if awr_root_stats:
            existing_awr_phvs = set()
            for s in batch.get('exec_stats_awr', []):
                phv = str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or '0')
                existing_awr_phvs.add(phv)
            new_rows = [s for s in awr_root_stats
                        if str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or '0') not in existing_awr_phvs]
            if new_rows:
                batch['exec_stats_awr'] = batch.get('exec_stats_awr', []) + new_rows
                logger.info("AWR root added %d new plan hash exec stats", len(new_rows))
        # Merge plan hashes from awr_root_sql_plan
        if awr_fb2.get('awr_plan_hashes'):
            batch['awr_plan_hashes'] = awr_fb2['awr_plan_hashes']

        # ── Unpack results ──────────────────────────────────────────────
        sql_text_rows = batch.get('sql_text', [])
        # AWR sql_text fallback: if cursor cache had nothing, use AWR text
        if not sql_text_rows:
            sql_text_rows = awr_fb2.get('sql_text_awr', []) or awr_fb2.get('sql_text_awr_root', [])
            if sql_text_rows:
                logger.info("SQL text retrieved from AWR (not in cursor cache)")

        # Combine cursor cache + AWR exec stats — MERGE both so we never lose
        # plan hashes that exist in AWR but not in cursor cache.
        exec_stats_cursor = batch.get('exec_stats_cursor', [])
        exec_stats_awr = batch.get('exec_stats_awr', [])

        # Filter out rows with null/zero plan_hash_value
        exec_stats_cursor = [s for s in exec_stats_cursor if _valid_phv(s)]
        exec_stats_awr = [s for s in exec_stats_awr if _valid_phv(s)]

        # Build merged exec_stats: cursor cache entries first (per-child detail),
        # then add AWR entries for any plan_hash_values not already covered.
        _seen_phvs = set()
        exec_stats = []
        for s in exec_stats_cursor:
            exec_stats.append(s)
            phv = str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or '0')
            _seen_phvs.add(phv)
        for s in exec_stats_awr:
            phv = str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or '0')
            if phv not in _seen_phvs:
                exec_stats.append(s)
                _seen_phvs.add(phv)

        # ── Execution plan — 3-layer fallback ──────────────────────────
        plan_rows = []
        plan_source = 'unavailable'
        raw_xplan_text = ''  # Preserve raw DBMS_XPLAN output for LLM

        # Layer 1: DBMS_XPLAN.DISPLAY_CURSOR
        cursor_text_rows = batch.get('plan_xplan', [])
        if cursor_text_rows:
            xplan_text = '\n'.join(
                str(r.get('PLAN_TABLE_OUTPUT') or r.get('plan_table_output') or '')
                for r in cursor_text_rows
            )
            parsed_rows = self._parse_xplan_text_to_rows(xplan_text)
            if parsed_rows:
                plan_rows = parsed_rows
                plan_source = 'DBMS_XPLAN (cursor cache)'
                raw_xplan_text = xplan_text  # Save only valid plan text

        # Layer 2: v$sql_plan (already in batch)
        if not plan_rows:
            plan_rows = batch.get('plan_vsqlplan', [])
            if plan_rows:
                plan_source = 'v$sql_plan'
                if not raw_xplan_text:
                    raw_xplan_text = self._format_structured_plan_as_text(
                        plan_rows, 'Shared Pool (v$sql_plan)')

        # Layer 3: awr_root_sql_plan — only fetch if layers 1+2 empty
        if not plan_rows:
            phv = None
            if exec_stats:
                phv_raw = exec_stats[0].get('PLAN_HASH_VALUE') or exec_stats[0].get('plan_hash_value')
                try:
                    phv = int(phv_raw) if phv_raw else None
                except (TypeError, ValueError):
                    phv = None

            fallback_queries: Dict[str, str] = {}
            if phv:
                fallback_queries["awrplan"] = (
                    f"SELECT id, parent_id, operation, options, object_owner, object_name, "
                    f"object_type, cost, cardinality, bytes, cpu_cost, io_cost, "
                    f"temp_space, access_predicates, filter_predicates, depth "
                    f"FROM awr_root_sql_plan "
                    f"WHERE sql_id = '{safe_id}' AND plan_hash_value = {phv} "
                    f"ORDER BY id "
                    f"FETCH FIRST 200 ROWS ONLY"
                )
            else:
                fallback_queries["awrplan"] = (
                    f"SELECT id, parent_id, operation, options, object_owner, object_name, "
                    f"object_type, cost, cardinality, bytes, cpu_cost, io_cost, "
                    f"temp_space, access_predicates, filter_predicates, depth "
                    f"FROM awr_root_sql_plan "
                    f"WHERE sql_id = '{safe_id}' "
                    f"AND plan_hash_value = ("
                    f"  SELECT MIN(plan_hash_value) FROM awr_root_sql_plan WHERE sql_id = '{safe_id}') "
                    f"ORDER BY id "
                    f"FETCH FIRST 200 ROWS ONLY"
                )

            if hasattr(self.conn, 'execute_batch_queries_dict'):
                fb = self.conn.execute_batch_queries_dict(fallback_queries)
            else:
                fb = {}
                for lbl, sql in fallback_queries.items():
                    try:
                        fb[lbl] = self.conn.execute_query_dict(sql) or []
                    except Exception:
                        fb[lbl] = []

            plan_rows = fb.get('awrplan', [])
            # Fallback: try awr_root_sql_plan / awr_pdb_sql_plan if awr_root_sql_plan returned nothing
            if not plan_rows:
                for alt_view in ('awr_root_sql_plan', 'awr_pdb_sql_plan'):
                    try:
                        retry_sql = fallback_queries["awrplan"].replace('awr_root_sql_plan', alt_view)
                        plan_rows = self.conn.execute_query_dict(retry_sql) or []
                        if plan_rows:
                            plan_source = f'{alt_view} (AWR)'
                            logger.info("Layer 3 plan from %s (%d rows)", alt_view, len(plan_rows))
                            break
                    except Exception:
                        continue
            if plan_rows and plan_source == 'unavailable':
                plan_source = 'awr_root_sql_plan (AWR)'
            if plan_rows and not raw_xplan_text:
                raw_xplan_text = self._format_structured_plan_as_text(
                    plan_rows, f'AWR History ({plan_source})')

        execution_history_daily = batch.get('execution_history_daily', [])

        # Fallback: if the primary awr_root_sqlstat query returned nothing, retry
        # awr_root and then the PDB-local AWR view (awr_pdb_sqlstat).
        if not execution_history_daily:
            _daily_fallback_views = [
                ('awr_root_sqlstat', 'awr_root_snapshot'),
                ('awr_pdb_sqlstat', 'awr_pdb_snapshot'),
            ]
            for _stat_view, _snap_view in _daily_fallback_views:
                try:
                    _daily_sql = (
                        f"SELECT * FROM ( "
                        f"  SELECT TO_CHAR(TRUNC(s.begin_interval_time), 'YYYY-MM-DD') AS exec_date, "
                        f"         t.plan_hash_value, "
                        f"         t.instance_number AS inst_id, "
                        f"         t.module, "
                        f"         SUM(t.executions_delta) AS execs, "
                        f"         SUM(t.buffer_gets_delta) AS buffer_gets, "
                        f"         SUM(t.rows_processed_delta) AS rows_processed, "
                        f"         ROUND(SUM(t.cpu_time_delta) / 1000000, 0) AS cpu_tim_secs, "
                        f"         ROUND(SUM(t.elapsed_time_delta) / 1000000, 0) AS ela_tim_secs, "
                        f"         ROUND(SUM(t.elapsed_time_delta) "
                        f"               / DECODE(SUM(t.executions_delta), 0, 1, SUM(t.executions_delta)) "
                        f"               / 1000000, 0) AS avg_ela_secs, "
                        f"         SUM(t.px_servers_execs_delta) AS px_tot "
                        f"  FROM {_stat_view} t "
                        f"  JOIN {_snap_view} s "
                        f"    ON t.snap_id = s.snap_id "
                        f"   AND t.instance_number = s.instance_number "
                        f"  WHERE t.sql_id = '{safe_id}' "
                        f"  GROUP BY TRUNC(s.begin_interval_time), "
                        f"           t.plan_hash_value, "
                        f"           t.instance_number, "
                        f"           t.module "
                        f"  ORDER BY TRUNC(s.begin_interval_time) DESC "
                        f") FETCH FIRST 40 ROWS ONLY"
                    )
                    _daily_rows = self.conn.execute_query_dict(_daily_sql) or []
                    if _daily_rows:
                        execution_history_daily = _daily_rows
                        logger.info("execution_history_daily: fetched %d rows via %s fallback",
                                    len(_daily_rows), _stat_view)
                        break
                except Exception as _ex:
                    logger.debug("execution_history_daily fallback [%s] failed: %s", _stat_view, _ex)

        ash_activity = batch.get('ash_activity', [])
        active_sessions = batch.get('active_sessions', [])
        session_wait_details = batch.get('session_wait_details', [])
        session_statistics = batch.get('session_statistics', [])
        transaction_rollback_info = batch.get('transaction_rollback', [])
        sort_usage_info = batch.get('sort_usage', [])
        sql_monitor = batch.get('sql_monitor', [])

        # ── Build all_plans: group v$sql_plan rows by child_number/plan_hash_value ──
        all_plan_rows = batch.get('all_plans_vsqlplan', [])
        plan_hash_list = batch.get('all_plan_hashes', [])
        all_plans_by_hash: Dict[str, Dict[str, Any]] = {}
        for row in all_plan_rows:
            child = str(row.get('CHILD_NUMBER') or row.get('child_number') or 0)
            phv = str(row.get('PLAN_HASH_VALUE') or row.get('plan_hash_value') or '0')
            key = f"{phv}_child{child}"
            if key not in all_plans_by_hash:
                all_plans_by_hash[key] = {
                    'plan_hash_value': phv,
                    'child_number': child,
                    'plan_rows': [],
                }
            all_plans_by_hash[key]['plan_rows'].append(row)

        # Always merge AWR historical plans so we never miss aged-out plans.
        # Collect known plan_hash_values from AWR exec stats AND awr_root_sql_plan
        known_phvs_awr = set()
        for s in exec_stats_awr:
            phv_raw = s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value')
            if phv_raw:
                try:
                    known_phvs_awr.add(int(phv_raw))
                except (TypeError, ValueError):
                    pass
        # Also include plan hashes found directly in awr_root_sql_plan
        for s in batch.get('awr_plan_hashes', []):
            phv_raw = s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value')
            if phv_raw:
                try:
                    known_phvs_awr.add(int(phv_raw))
                except (TypeError, ValueError):
                    pass

        # Also collect plan hashes already present from cursor cache
        cursor_phvs = set()
        for entry in all_plans_by_hash.values():
            try:
                cursor_phvs.add(int(entry['plan_hash_value']))
            except (TypeError, ValueError):
                pass

        # Determine which AWR plan hashes we still need to fetch
        missing_phvs = known_phvs_awr - cursor_phvs

        awr_all_plans_queries: Dict[str, str] = {}
        if missing_phvs:
            # Fetch only plan hashes not already in cursor cache
            phv_list = ','.join(str(p) for p in missing_phvs)
            awr_all_plans_queries["awr_all_plans"] = (
                f"SELECT plan_hash_value, "
                f"id, parent_id, operation, options, object_owner, object_name, "
                f"object_type, cost, cardinality, bytes, cpu_cost, io_cost, "
                f"temp_space, access_predicates, filter_predicates, depth "
                f"FROM awr_root_sql_plan WHERE sql_id = '{safe_id}' "
                f"AND plan_hash_value IN ({phv_list}) "
                f"ORDER BY plan_hash_value, id "
                f"FETCH FIRST 500 ROWS ONLY"
            )
        elif not all_plans_by_hash:
            # Cursor cache was completely empty — fetch all AWR plans
            awr_all_plans_queries["awr_all_plans"] = (
                f"SELECT plan_hash_value, "
                f"id, parent_id, operation, options, object_owner, object_name, "
                f"object_type, cost, cardinality, bytes, cpu_cost, io_cost, "
                f"temp_space, access_predicates, filter_predicates, depth "
                f"FROM awr_root_sql_plan WHERE sql_id = '{safe_id}' "
                f"ORDER BY plan_hash_value, id "
                f"FETCH FIRST 500 ROWS ONLY"
            )

        if awr_all_plans_queries:
            if hasattr(self.conn, 'execute_batch_queries_dict'):
                awr_fb = self.conn.execute_batch_queries_dict(awr_all_plans_queries)
            else:
                awr_fb = {}
                for lbl, sql in awr_all_plans_queries.items():
                    try:
                        awr_fb[lbl] = self.conn.execute_query_dict(sql) or []
                    except Exception:
                        awr_fb[lbl] = []
            # Fallback: if awr_root_sql_plan returned nothing, try awr_root_sql_plan
            if not awr_fb.get('awr_all_plans'):
                for view_name in ('awr_root_sql_plan', 'awr_pdb_sql_plan'):
                    retry_sql = awr_all_plans_queries["awr_all_plans"].replace('awr_root_sql_plan', view_name)
                    try:
                        retry_rows = self.conn.execute_query_dict(retry_sql) if hasattr(self.conn, 'execute_query_dict') else []
                        if retry_rows:
                            awr_fb['awr_all_plans'] = retry_rows
                            logger.info("AWR all_plans fetched from %s (%d rows)", view_name, len(retry_rows))
                            break
                    except Exception:
                        continue
            for row in awr_fb.get('awr_all_plans', []):
                phv = str(row.get('PLAN_HASH_VALUE') or row.get('plan_hash_value') or '0')
                key = f"{phv}_awr"
                if key not in all_plans_by_hash:
                    all_plans_by_hash[key] = {
                        'plan_hash_value': phv,
                        'child_number': 'AWR',
                        'plan_rows': [],
                    }
                all_plans_by_hash[key]['plan_rows'].append(row)

        # Also enrich with exec stats per plan hash from both sources
        exec_stats_by_phv: Dict[str, Dict[str, Any]] = {}
        for s in exec_stats_cursor:
            phv = str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or '0')
            if phv not in exec_stats_by_phv:
                exec_stats_by_phv[phv] = {
                    'source': 'cursor_cache',
                    'executions': 0, 'elapsed_time': 0, 'cpu_time': 0,
                    'buffer_gets': 0, 'disk_reads': 0, 'rows_processed': 0,
                    'last_active_time': '', 'first_load_time': '',
                }
            st = exec_stats_by_phv[phv]
            st['executions'] += int(s.get('EXECUTIONS') or s.get('executions') or 0)
            st['elapsed_time'] += int(s.get('ELAPSED_TIME') or s.get('elapsed_time') or 0)
            st['cpu_time'] += int(s.get('CPU_TIME') or s.get('cpu_time') or 0)
            st['buffer_gets'] += int(s.get('BUFFER_GETS') or s.get('buffer_gets') or 0)
            st['disk_reads'] += int(s.get('DISK_READS') or s.get('disk_reads') or 0)
            st['rows_processed'] += int(s.get('ROWS_PROCESSED') or s.get('rows_processed') or 0)
            # Track the most recent last_active_time for this plan hash
            lat = str(s.get('LAST_ACTIVE_TIME') or s.get('last_active_time') or '')
            if lat and lat > st['last_active_time']:
                st['last_active_time'] = lat
            flt = str(s.get('FIRST_LOAD_TIME') or s.get('first_load_time') or '')
            if flt and (not st['first_load_time'] or flt < st['first_load_time']):
                st['first_load_time'] = flt
        for s in exec_stats_awr:
            phv = str(s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value') or '0')
            if phv not in exec_stats_by_phv:
                exec_stats_by_phv[phv] = {
                    'source': 'AWR',
                    'executions': 0, 'elapsed_time': 0, 'cpu_time': 0,
                    'buffer_gets': 0, 'disk_reads': 0, 'rows_processed': 0,
                    'last_active_time': '', 'first_load_time': '',
                }
            st = exec_stats_by_phv[phv]
            st['executions'] += int(s.get('EXECUTIONS') or s.get('executions') or 0)
            st['elapsed_time'] += int(s.get('ELAPSED_TIME') or s.get('elapsed_time') or 0)
            st['cpu_time'] += int(s.get('CPU_TIME') or s.get('cpu_time') or 0)
            st['buffer_gets'] += int(s.get('BUFFER_GETS') or s.get('buffer_gets') or 0)
            st['disk_reads'] += int(s.get('DISK_READS') or s.get('disk_reads') or 0)
            st['rows_processed'] += int(s.get('ROWS_PROCESSED') or s.get('rows_processed') or 0)
            # Track most-recent last_active_time (MAX snapshot end time) for AWR plans
            lat = str(s.get('LAST_ACTIVE_TIME') or s.get('last_active_time') or '')
            if lat and lat > st['last_active_time']:
                st['last_active_time'] = lat

        # Attach per-plan-hash aggregated stats to each plan entry
        for entry in all_plans_by_hash.values():
            phv = entry['plan_hash_value']
            if phv in exec_stats_by_phv:
                st = exec_stats_by_phv[phv]
                execs = st['executions'] or 1
                entry['exec_stats'] = {
                    'executions': st['executions'],
                    'avg_elapsed_ms': round(st['elapsed_time'] / execs / 1000, 2),
                    'avg_cpu_ms': round(st['cpu_time'] / execs / 1000, 2),
                    'avg_buffer_gets': round(st['buffer_gets'] / execs, 1),
                    'avg_disk_reads': round(st['disk_reads'] / execs, 1),
                    'avg_rows': round(st['rows_processed'] / execs, 1),
                    'last_active_time': st.get('last_active_time', ''),
                    'source': st['source'],
                }

        # Ensure every discovered plan hash is represented, even when plan rows are unavailable.
        known_plan_hashes = set()
        for row in plan_hash_list:
            phv_raw = row.get('PLAN_HASH_VALUE') or row.get('plan_hash_value')
            try:
                phv_int = int(phv_raw)
                if phv_int > 0:
                    known_plan_hashes.add(phv_int)
            except (TypeError, ValueError):
                continue
        for row in batch.get('awr_plan_hashes', []):
            phv_raw = row.get('PLAN_HASH_VALUE') or row.get('plan_hash_value')
            try:
                phv_int = int(phv_raw)
                if phv_int > 0:
                    known_plan_hashes.add(phv_int)
            except (TypeError, ValueError):
                continue
        for s in exec_stats:
            phv_raw = s.get('PLAN_HASH_VALUE') or s.get('plan_hash_value')
            try:
                phv_int = int(phv_raw)
                if phv_int > 0:
                    known_plan_hashes.add(phv_int)
            except (TypeError, ValueError):
                continue

        for phv_int in sorted(known_plan_hashes):
            phv = str(phv_int)
            exists = any(entry.get('plan_hash_value') == phv for entry in all_plans_by_hash.values())
            if exists:
                continue
            st = exec_stats_by_phv.get(phv, {})
            execs = int(st.get('executions') or 0)
            denom = max(execs, 1)
            all_plans_by_hash[f"{phv}_known"] = {
                'plan_hash_value': phv,
                'child_number': 'N/A',
                'plan_rows': [],
                'exec_stats': {
                    'executions': execs,
                    'avg_elapsed_ms': round(float(st.get('elapsed_time') or 0) / denom / 1000, 2),
                    'avg_cpu_ms': round(float(st.get('cpu_time') or 0) / denom / 1000, 2),
                    'avg_buffer_gets': round(float(st.get('buffer_gets') or 0) / denom, 1),
                    'avg_disk_reads': round(float(st.get('disk_reads') or 0) / denom, 1),
                    'avg_rows': round(float(st.get('rows_processed') or 0) / denom, 1),
                    'source': st.get('source', 'AWR'),
                },
            }

        # Ensure every plan_hash that has exec_stats also has an all_plans entry
        # (even if no plan_rows were found — so the UI at least shows the plan hash
        #  with its exec stats)
        for phv, st in exec_stats_by_phv.items():
            if phv == '0':
                continue
            # Check if this phv already exists in all_plans_by_hash
            found = any(entry['plan_hash_value'] == phv for entry in all_plans_by_hash.values())
            if not found:
                key = f"{phv}_stats"
                execs = st['executions'] or 1
                all_plans_by_hash[key] = {
                    'plan_hash_value': phv,
                    'child_number': 'AWR',
                    'plan_rows': [],
                    'exec_stats': {
                        'executions': st['executions'],
                        'avg_elapsed_ms': round(st['elapsed_time'] / execs / 1000, 2),
                        'avg_cpu_ms': round(st['cpu_time'] / execs / 1000, 2),
                        'avg_buffer_gets': round(st['buffer_gets'] / execs, 1),
                        'avg_disk_reads': round(st['disk_reads'] / execs, 1),
                        'avg_rows': round(st['rows_processed'] / execs, 1),
                        'source': st['source'],
                    },
                }

        all_plans = list(all_plans_by_hash.values())

        # Alternate plan detection
        sig_rows = batch.get('signature', [])
        force_sig = None
        current_plan_hash = None
        current_avg_sec = 0.0
        if sig_rows:
            try:
                force_sig = int(sig_rows[0].get('FORCE_MATCHING_SIGNATURE') or sig_rows[0].get('force_matching_signature') or 0)
            except (TypeError, ValueError):
                force_sig = None
            try:
                current_plan_hash = int(sig_rows[0].get('PLAN_HASH_VALUE') or sig_rows[0].get('plan_hash_value') or 0)
            except (TypeError, ValueError):
                current_plan_hash = None
            try:
                current_avg_sec = float(sig_rows[0].get('AVG_ELAPSED_SEC') or sig_rows[0].get('avg_elapsed_sec') or 0)
            except (TypeError, ValueError):
                current_avg_sec = 0.0

        # AWR fallback: if cursor cache had no current plan, derive from exec_stats
        if current_plan_hash is None and exec_stats:
            # Pick the plan with most executions as "current"
            best_row = max(exec_stats, key=lambda s: int(s.get('EXECUTIONS') or s.get('executions') or 0))
            try:
                current_plan_hash = int(best_row.get('PLAN_HASH_VALUE') or best_row.get('plan_hash_value') or 0)
                execs = int(best_row.get('EXECUTIONS') or best_row.get('executions') or 1)
                elapsed = float(best_row.get('ELAPSED_TIME') or best_row.get('elapsed_time') or 0)
                if execs > 0 and elapsed > 0:
                    current_avg_sec = elapsed / execs / 1e6
                else:
                    current_avg_sec = float(best_row.get('AVG_ELAPSED_MS') or best_row.get('avg_elapsed_ms') or 0) / 1000.0
            except (TypeError, ValueError):
                pass

        alternate_plans = self._collect_oracle_alternate_plans(safe_id, current_plan_hash, force_sig)

        # Fallback: if alternate_plans is empty but we have multiple plan hashes
        # in exec_stats, build alternate_plans from exec_stats data.
        if not alternate_plans and len(exec_stats_by_phv) > 1:
            for phv, st in exec_stats_by_phv.items():
                if phv == '0':
                    continue
                execs = st['executions'] or 1
                try:
                    phv_int = int(phv)
                except (TypeError, ValueError):
                    continue
                alternate_plans.append({
                    'sql_id': safe_id,
                    'plan_hash_value': phv_int,
                    'executions': st['executions'],
                    'avg_elapsed_sec': round(st['elapsed_time'] / execs / 1e6, 4),
                    'avg_cpu_sec': round(st['cpu_time'] / execs / 1e6, 4),
                    'last_active_time': st.get('last_active_time', ''),
                    'first_load_time': st.get('first_load_time', ''),
                    'is_current_plan': bool(current_plan_hash is not None and phv_int == current_plan_hash),
                    'source': st['source'],
                })

        # Ensure ALL known plans appear in alternate_plans — merge any from
        # all_plans that are missing (e.g. plans in awr_root_sql_plan with
        # 0 executions, or plans only found via plan hash list).
        # Normalize to int to avoid str/int mismatch.
        alt_phvs: set = set()
        for p in alternate_plans:
            try:
                alt_phvs.add(int(p['plan_hash_value']))
            except (TypeError, ValueError, KeyError):
                pass
        logger.info("Merge block: alt_phvs=%s, all_plans count=%d", alt_phvs, len(all_plans))
        for entry in all_plans:
            phv_raw = entry.get('plan_hash_value')
            if not phv_raw:
                continue
            try:
                phv_int = int(phv_raw)
            except (TypeError, ValueError):
                continue
            if phv_int in alt_phvs:
                continue
            es = entry.get('exec_stats', {})
            # Get last_active_time from exec_stats_by_phv if available
            phv_str = str(phv_int)
            phv_times = exec_stats_by_phv.get(phv_str, {})
            alternate_plans.append({
                'sql_id': safe_id,
                'plan_hash_value': phv_int,
                'executions': es.get('executions', 0),
                'avg_elapsed_sec': round(es.get('avg_elapsed_ms', 0) / 1000.0, 4) if es.get('avg_elapsed_ms') else 0,
                'avg_cpu_sec': round(es.get('avg_cpu_ms', 0) / 1000.0, 4) if es.get('avg_cpu_ms') else 0,
                'last_active_time': phv_times.get('last_active_time', ''),
                'first_load_time': phv_times.get('first_load_time', ''),
                'is_current_plan': bool(current_plan_hash is not None and phv_int == current_plan_hash),
                'source': es.get('source', entry.get('child_number', 'unknown')),
                'plan_rows_count': len(entry.get('plan_rows', [])),
            })
            alt_phvs.add(phv_int)
        logger.info("Merge block done: alternate_plans now %d entries", len(alternate_plans))

        plan_comparison = self._compare_oracle_plans(
            current_avg_sec=current_avg_sec,
            current_plan_hash=current_plan_hash,
            current_executions=int((exec_stats[0].get('EXECUTIONS') or exec_stats[0].get('executions') or 0) if exec_stats else 0),
            alternate_plans=alternate_plans,
        )

        best_plan_hash_value = plan_comparison.get('best_plan_hash_value')
        current_plan_hash_value = plan_comparison.get('current_plan_hash_value')
        best_plan_hash_text = str(best_plan_hash_value) if best_plan_hash_value is not None else ''
        current_plan_hash_text = str(current_plan_hash_value) if current_plan_hash_value is not None else ''
        for p in all_plans:
            phv = str(p.get('plan_hash_value') or '')
            p['is_best_plan'] = bool(best_plan_hash_text and phv == best_plan_hash_text)
            p['is_current_plan'] = bool(current_plan_hash_text and phv == current_plan_hash_text)

        available_plan_hash_values = sorted({str(v) for v in known_plan_hashes}, key=lambda x: int(x))

        # ─────── Generate SPECIFIC recommendations (no generic hints) ───────
        recommendations = []
        plan_analysis = {}
        cardinality_errors = []
        bind_skew = {'has_skew': False}

        if plan_rows:
            # Structured plan analysis
            plan_analysis = self._analyze_plan_structure_foolproof(plan_rows)

            # Try to detect cardinality errors if we have display_cursor plan
            try:
                # Reuse DISPLAY_CURSOR data already fetched in the batch
                cursor_plan_rows = batch.get('plan_xplan', [])
                cursor_plan_text = '\n'.join(
                    str(r.get('PLAN_TABLE_OUTPUT') or r.get('plan_table_output') or '')
                    for r in cursor_plan_rows
                )
                cardinality_errors = self._detect_cardinality_errors(cursor_plan_text, plan_rows)
            except Exception:
                pass

        # Detect bind variable skew
        bind_skew = self._detect_bind_variable_skew(safe_id)

        # Generate SPECIFIC recommendations from analysis
        recommendations = self._generate_specific_recommendations(
            plan_analysis,
            cardinality_errors,
            bind_skew,
            exec_stats
        )

        # Add SPM / plan baseline recommendation when better plan is detected
        if plan_comparison.get('better_plan_found'):
            best_phv = plan_comparison.get('best_plan_hash_value')
            cur_phv = plan_comparison.get('current_plan_hash_value')
            improvement = plan_comparison.get('estimated_improvement_pct', 0)
            confidence = plan_comparison.get('confidence_pct', 0)

            # Determine source of the best plan to recommend the right approach
            best_source = 'AWR'
            for ap in alternate_plans:
                if ap.get('plan_hash_value') == best_phv:
                    best_source = ap.get('source', 'AWR')
                    break

            # Build source-appropriate action scripts
            if best_source == 'cursor_cache':
                baseline_action = (
                    f"-- Option 1: SQL Plan Baseline from cursor cache (no Tuning Pack required)\n"
                    f"DECLARE\n"
                    f"  l_plans PLS_INTEGER;\n"
                    f"BEGIN\n"
                    f"  l_plans := DBMS_SPM.LOAD_PLANS_FROM_CURSOR_CACHE(\n"
                    f"    sql_id          => '{safe_id}',\n"
                    f"    plan_hash_value => {best_phv},\n"
                    f"    fixed           => 'YES',\n"
                    f"    enabled         => 'YES');\n"
                    f"  DBMS_OUTPUT.PUT_LINE('Plans loaded: ' || l_plans);\n"
                    f"END;\n"
                    f"/\n"
                    f"-- Verify:\n"
                    f"SELECT sql_handle, plan_name, enabled, fixed, accepted\n"
                    f"  FROM DBA_SQL_PLAN_BASELINES\n"
                    f" WHERE DBMS_SPM.GET_SQL_PLAN_BASELINE_ATTRIBUTE(sql_handle, plan_name, 'SQL_ID') = '{safe_id}';"
                )
                profile_action = (
                    f"-- Option 2: SQL Profile (requires Tuning Pack license)\n"
                    f"-- Use 'Generate Plan Fix Script' button to create a full SQL Profile script\n"
                    f"-- that extracts optimizer hints from the plan and works across systems."
                )
            else:
                baseline_action = (
                    f"-- Option 1: SQL Plan Baseline from AWR\n"
                    f"DECLARE\n"
                    f"  l_plans PLS_INTEGER;\n"
                    f"BEGIN\n"
                    f"  l_plans := DBMS_SPM.LOAD_PLANS_FROM_AWR(\n"
                    f"    begin_snap   => (SELECT MIN(snap_id) FROM awr_root_snapshot\n"
                    f"                     WHERE begin_interval_time > SYSDATE - 30),\n"
                    f"    end_snap     => (SELECT MAX(snap_id) FROM awr_root_snapshot),\n"
                    f"    basic_filter => 'sql_id = ''{safe_id}''\n"
                    f"                     AND plan_hash_value = {best_phv}');\n"
                    f"  DBMS_OUTPUT.PUT_LINE('Plans loaded: ' || l_plans);\n"
                    f"END;\n"
                    f"/\n"
                    f"-- Verify:\n"
                    f"SELECT sql_handle, plan_name, enabled, fixed, accepted\n"
                    f"  FROM DBA_SQL_PLAN_BASELINES\n"
                    f" WHERE DBMS_SPM.GET_SQL_PLAN_BASELINE_ATTRIBUTE(sql_handle, plan_name, 'SQL_ID') = '{safe_id}';"
                )
                profile_action = (
                    f"-- Option 2: SQL Profile from AWR (requires Tuning Pack license)\n"
                    f"-- Use 'Generate Plan Fix Script' button to create a full SQL Profile script\n"
                    f"-- that extracts optimizer hints from AWR and can be transferred across systems."
                )

            recommendations.append({
                'type': 'PLAN_STABILITY',
                'severity': 'HIGH' if confidence >= 60 else 'MEDIUM',
                'title': f'Better execution plan detected — plan hash {best_phv} is {improvement:.0f}% faster',
                'evidence': {
                    'current_plan_hash': cur_phv,
                    'best_plan_hash': best_phv,
                    'improvement_pct': improvement,
                    'confidence_pct': confidence,
                    'current_avg_sec': plan_comparison.get('current_avg_elapsed_sec'),
                    'best_avg_sec': plan_comparison.get('best_avg_elapsed_sec'),
                    'best_plan_source': best_source,
                },
                'expected_benefit': f'Reduce average elapsed time by ~{improvement:.0f}% by pinning the optimal plan',
                'risk_level': 'LOW',
                'action': baseline_action + "\n\n" + profile_action,
            })

        existing_indexes = self._collect_existing_indexes_for_plan(plan_rows)

        # ── Collect table, column, and index statistics for plan objects ──
        predicate_columns = []
        pred_regex = re.compile(r'([A-Z][A-Z0-9_#$]{1,30})\s*(?:=|<|>|<=|>=|LIKE|IN|BETWEEN)', re.IGNORECASE)
        for row in plan_rows:
            if not isinstance(row, dict):
                continue
            for pred_key in ('ACCESS_PREDICATES', 'access_predicates', 'FILTER_PREDICATES', 'filter_predicates'):
                pred = row.get(pred_key)
                if not pred:
                    continue
                for m in pred_regex.finditer(str(pred)):
                    col = m.group(1).upper()
                    if col not in predicate_columns:
                        predicate_columns.append(col)
        object_statistics = self._collect_object_stats_for_plan(plan_rows, predicate_columns)

        return {
            'database_type': 'oracle',
            'sql_id': safe_id,
            'sql_text': sql_text_rows,
            # "Current Execution Statistics (v$sql)" — cursor cache only.
            # Historical AWR data is surfaced separately via execution_history_daily.
            'execution_stats': exec_stats_cursor,
            'execution_history_daily': execution_history_daily,
            'execution_plan': plan_rows,
            'execution_plan_rows': plan_rows,
            'execution_plan_text': raw_xplan_text,
            'plan_source': plan_source,
            'current_plan_hash_value': current_plan_hash,
            'best_plan_hash_value': plan_comparison.get('best_plan_hash_value'),
            'available_plan_hash_values': available_plan_hash_values,
            'all_plans': all_plans,
            'plan_analysis': plan_analysis,
            'cardinality_errors': cardinality_errors,
            'bind_variable_skew': bind_skew,
            'existing_indexes': existing_indexes,
            'object_statistics': object_statistics,
            'alternate_plans': alternate_plans,
            'plan_comparison': plan_comparison,
            'recommendations': recommendations,
            'specific_recommendations': recommendations,
            'ash_activity': ash_activity,
            'active_sessions': active_sessions,
            'session_wait_details': session_wait_details,
            'session_statistics': session_statistics,
            'transaction_rollback_info': transaction_rollback_info,
            'sort_usage_info': sort_usage_info,
            'sql_monitor': sql_monitor,
            'bind_variables': [],
        }

    def _collect_pg_sqlid_info(self, query_id: str) -> Dict[str, Any]:
        """Collect PostgreSQL query details from pg_stat_statements."""
        safe_id = str(query_id).strip()[:30]

        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("PG query ID lookup failed: %s", ex)
                return []

        # pg_stat_statements entry
        stats = _q(
            f"SELECT queryid, LEFT(query, 2000) AS query_text, "
            f"calls, total_exec_time, mean_exec_time, min_exec_time, max_exec_time, "
            f"stddev_exec_time, rows, shared_blks_hit, shared_blks_read, "
            f"shared_blks_written, temp_blks_read, temp_blks_written "
            f"FROM pg_stat_statements WHERE queryid::text = '{safe_id}' LIMIT 1"
        )

        # Active sessions running this query
        active = _q(
            f"SELECT pid, usename, application_name, client_addr, "
            f"state, wait_event_type, wait_event, "
            f"LEFT(query, 500) AS query_text, "
            f"EXTRACT(EPOCH FROM (now() - query_start))::INT AS elapsed_seconds "
            f"FROM pg_stat_activity "
            f"WHERE query LIKE '%' || LEFT((SELECT query FROM pg_stat_statements WHERE queryid::text = '{safe_id}' LIMIT 1), 50) || '%' "
            f"AND pid != pg_backend_pid() LIMIT 10"
        )

        return {
            'database_type': 'postgresql',
            'sql_id': safe_id,
            'sql_text': stats,
            'execution_stats': stats,
            'execution_plan': [],
            'ash_activity': [],
            'active_sessions': active,
            'session_wait_details': [],
            'session_statistics': [],
            'transaction_rollback_info': [],
            'sort_usage_info': [],
            'sql_monitor': [],
            'bind_variables': [],
        }

    def collect_session_info(self, sid: int, serial: int = 0) -> Dict[str, Any]:
        """Collect comprehensive session details by SID (Oracle) or PID (PostgreSQL)."""
        db_type = self.conn.get_database_type().lower()
        if db_type == 'oracle':
            return self._collect_oracle_session_info(sid, serial)
        return self._collect_pg_session_info(sid)

    def _collect_oracle_session_info(self, sid: int, serial: int = 0) -> Dict[str, Any]:
        """Collect Oracle session details from V$SESSION and related views."""
        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("Session info query failed: %s", ex)
                return []

        serial_clause = f"AND s.serial# = {serial}" if serial else ""

        # Session details
        session_info = _q(
            f"SELECT s.sid, s.serial#, s.username, s.machine, s.program, "
            f"s.module, s.action, s.status, s.logon_time, s.type, "
            f"s.event, s.wait_class, s.state, "
            f"ROUND(s.seconds_in_wait) AS seconds_waiting, "
            f"s.sql_id, s.prev_sql_id, s.blocking_session, "
            f"SUBSTR(sq.sql_text, 1, 500) AS current_sql, "
            f"SUBSTR(psq.sql_text, 1, 500) AS prev_sql "
            f"FROM v$session s "
            f"LEFT JOIN v$sql sq ON sq.sql_id = s.sql_id AND sq.child_number = s.sql_child_number "
            f"LEFT JOIN v$sql psq ON psq.sql_id = s.prev_sql_id AND psq.child_number = 0 "
            f"WHERE s.sid = {sid} {serial_clause}"
        )

        # Session statistics (top resource consumers)
        session_stats = _q(
            f"SELECT n.name, ss.value "
            f"FROM v$sesstat ss JOIN v$statname n ON n.statistic# = ss.statistic# "
            f"WHERE ss.sid = {sid} "
            f"AND n.name IN ('session logical reads', 'physical reads', 'db block changes', "
            f"'redo size', 'parse count (hard)', 'parse count (total)', "
            f"'user commits', 'user rollbacks', 'execute count', 'session pga memory') "
            f"ORDER BY ss.value DESC"
        )

        # Session wait history
        wait_history = _q(
            f"SELECT event, wait_class, "
            f"ROUND(time_waited_micro/1000, 2) AS time_waited_ms "
            f"FROM v$session_event WHERE sid = {sid} "
            f"AND wait_class != 'Idle' "
            f"ORDER BY time_waited_micro DESC FETCH FIRST 15 ROWS ONLY"
        )

        # Session's open cursors
        open_cursors = _q(
            f"SELECT sql_id, sql_text, cursor_type "
            f"FROM v$open_cursor WHERE sid = {sid} "
            f"AND ROWNUM <= 10"
        )

        # ASH activity for this session (last 30 min)
        ash_for_session = _q(
            f"SELECT sql_id, NVL(event, 'On CPU') AS event, wait_class, "
            f"COUNT(*) AS sample_count "
            f"FROM v$active_session_history "
            f"WHERE session_id = {sid} AND sample_time > SYSDATE - 30/1440 "
            f"GROUP BY sql_id, event, wait_class "
            f"ORDER BY sample_count DESC FETCH FIRST 15 ROWS ONLY"
        )

        # Who is this session blocking?
        blocked_sessions = _q(
            f"SELECT sid, serial#, username, event, "
            f"ROUND(seconds_in_wait) AS seconds_waiting, sql_id "
            f"FROM v$session WHERE blocking_session = {sid}"
        )

        return {
            'database_type': 'oracle',
            'sid': sid,
            'serial': serial,
            'session_info': session_info,
            'session_stats': session_stats,
            'wait_history': wait_history,
            'open_cursors': open_cursors,
            'ash_activity': ash_for_session,
            'blocked_sessions': blocked_sessions,
        }

    def _collect_pg_session_info(self, pid: int) -> Dict[str, Any]:
        """Collect PostgreSQL session/process details."""
        def _q(sql):
            try:
                return self.conn.execute_query_dict(sql) or []
            except Exception as ex:
                logger.warning("PG session info query failed: %s", ex)
                return []

        session_info = _q(
            f"SELECT pid, usename, application_name, client_addr, client_port, "
            f"backend_start, xact_start, query_start, state_change, "
            f"state, wait_event_type, wait_event, "
            f"LEFT(query, 1000) AS current_sql, "
            f"backend_type "
            f"FROM pg_stat_activity WHERE pid = {pid}"
        )

        # Locks held by this PID
        locks = _q(
            f"SELECT locktype, relation::regclass AS relation, mode, granted, "
            f"page, tuple "
            f"FROM pg_locks WHERE pid = {pid} LIMIT 20"
        )

        # Who is this session blocking?
        blocked = _q(
            f"SELECT pid, usename, state, wait_event, "
            f"LEFT(query, 200) AS query_text, "
            f"EXTRACT(EPOCH FROM (now() - query_start))::INT AS elapsed_seconds "
            f"FROM pg_stat_activity "
            f"WHERE cardinality(pg_blocking_pids(pid)) > 0 "
            f"AND {pid} = ANY(pg_blocking_pids(pid))"
        )

        return {
            'database_type': 'postgresql',
            'sid': pid,
            'serial': 0,
            'session_info': session_info,
            'session_stats': [],
            'wait_history': [],
            'open_cursors': [],
            'ash_activity': [],
            'blocked_sessions': blocked,
            'locks': locks,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Batched Oracle metrics — runs ALL queries in 1 JVM session
    # ──────────────────────────────────────────────────────────────────────

    def _collect_all_oracle_metrics_batched(self) -> Dict[str, Any]:
        """Collect all Oracle metric categories via a single SQLcl/JVM session.

        Instead of spawning ~17 JVM processes (one per query), this method
        sends all queries to ``execute_batch_queries_dict`` which SPOOLs
        each result into a separate temp file — 1 JVM, ~17 queries.
        """
        import time as _time

        _t0 = _time.monotonic()

        def _scalar(value: Any) -> Any:
            if isinstance(value, (list, tuple)):
                if not value:
                    return 0
                return _scalar(value[0])
            return value

        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                v = _scalar(value)
                if v in (None, ''):
                    return default
                return int(v)
            except (TypeError, ValueError):
                return default

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                v = _scalar(value)
                if v in (None, ''):
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        # Build the batch of all queries keyed by a unique label.
        # NOTE: SQLcl 19.1 has a script buffer limit — queries at the end of
        # very large batches may never execute.  Put small, critical queries
        # FIRST so they always run.
        queries: dict = {
            # ── live metrics (FIRST — small & critical for health report) ──
            "live_db_identity": "SELECT name AS db_name, dbid, open_mode FROM v$database",
            "live_db_identity_fallback": (
                "SELECT SYS_CONTEXT('USERENV','DB_NAME') AS db_name, "
                "SYS_CONTEXT('USERENV','DB_UNIQUE_NAME') AS db_unique_name "
                "FROM dual"
            ),
            "live_instance_status": (
                "SELECT status, instance_name, host_name FROM v$instance"
            ),
            "live_total_sess": "SELECT COUNT(*) AS total_sessions FROM v$session",
            "live_max_proc": "SELECT value FROM v$parameter WHERE name = 'processes'",
            "live_os_stats": (
                "SELECT stat_name, value FROM v$osstat "
                "WHERE stat_name IN ('PHYSICAL_MEMORY_BYTES','FREE_MEMORY_BYTES','NUM_CPUS','SYS_TIME','IDLE_TIME','USER_TIME')"
            ),
            "live_db_cache_sz": "SELECT value AS db_cache_size FROM v$parameter WHERE name='db_cache_size'",
            "live_redo_wait": "SELECT value AS redo_space_requests FROM v$sysstat WHERE name='redo log space requests'",
            "live_lib_cache": (
                "SELECT ROUND(SUM(pinhits)/NULLIF(SUM(pins),0)*100, 2) AS lib_cache_hit_pct "
                "FROM v$librarycache"
            ),
            "live_sga": "SELECT name, ROUND(value/1024/1024,1) AS value_mb FROM v$sga",
            "live_pga": (
                "SELECT name, ROUND(value/1024/1024,1) AS value_mb FROM v$pgastat "
                "WHERE name IN ('aggregate PGA target parameter','aggregate PGA auto target',"
                "'total PGA inuse','total PGA allocated','over allocation count')"
            ),
            "live_wait_events": (
                "SELECT event, total_waits, time_waited FROM v$system_event "
                "WHERE wait_class != 'Idle' ORDER BY time_waited DESC FETCH FIRST 10 ROWS ONLY"
            ),
            "live_unusable_cnt": "SELECT COUNT(*) AS unusable_indexes FROM dba_indexes WHERE status='UNUSABLE'",
            "live_unusable_det": (
                "SELECT owner, index_name, table_name, status "
                "FROM dba_indexes WHERE status='UNUSABLE' "
                "ORDER BY owner, index_name FETCH FIRST 25 ROWS ONLY"
            ),
            "live_invalid_cnt": "SELECT COUNT(*) AS invalid_objects FROM dba_objects WHERE status='INVALID'",
            "live_dblink_all": (
                "SELECT owner, db_link, host, created FROM dba_db_links ORDER BY db_link"
            ),
            "live_dblink_status": (
                "SELECT db_link, owner, host, valid, created "
                "FROM dba_db_links ORDER BY db_link"
            ),
            # ── general ──
            "gen_version": (
                "SELECT banner FROM v$version WHERE banner LIKE 'Oracle Database%' FETCH FIRST 1 ROWS ONLY"
            ),
            "gen_uptime": (
                "SELECT (sysdate - startup_time)*24*60*60 AS uptime_seconds FROM v$instance"
            ),
            "gen_dbsize": (
                "SELECT NVL(SUM(bytes), 0) AS total_bytes FROM dba_data_files"
            ),
            # ── connections ──
            "conn_by_status": (
                "SELECT status, COUNT(*) AS connection_count FROM v$session GROUP BY status"
            ),
            "conn_active": (
                "SELECT COUNT(*) AS active_connections FROM v$session WHERE status='ACTIVE'"
            ),
            "conn_max": (
                "SELECT value FROM v$parameter WHERE name='sessions'"
            ),
            # ── cache ──
            "cache_hit": (
                "SELECT ROUND(1 - (SUM(CASE name WHEN 'physical reads' THEN value ELSE 0 END) / "
                "NULLIF(SUM(CASE name WHEN 'db block gets' THEN value "
                "WHEN 'consistent gets' THEN value ELSE 0 END), 0)), 4) * 100 AS hit_ratio "
                "FROM v$sysstat WHERE name IN ('db block gets', 'consistent gets', 'physical reads')"
            ),
            # ── queries (unified weighted scoring, SYS-filtered) ──
            "query_slow": _oracle_top_sql_for_batch(20),
            "query_total": (
                "SELECT COUNT(*) AS total_queries FROM v$sql"
            ),
            # ── indexes ──
            "idx_unused": (
                "SELECT owner, name AS index_name, total_access_count, total_rows_returned "
                "FROM dba_index_usage WHERE total_access_count = 0 FETCH FIRST 20 ROWS ONLY"
            ),
            "idx_large": (
                "SELECT owner, segment_name AS index_name, bytes "
                "FROM dba_segments WHERE segment_type='INDEX' "
                "ORDER BY bytes DESC FETCH FIRST 20 ROWS ONLY"
            ),
            "idx_frag": (
                "SELECT owner, index_name, table_name, status, blevel, leaf_blocks, num_rows, clustering_factor, "
                "ROUND(CASE WHEN NVL(num_rows, 0) = 0 THEN 0 "
                "           ELSE (clustering_factor / NULLIF(num_rows, 0)) * 100 END, 2) AS fragmentation_pct "
                "FROM dba_indexes "
                "WHERE status = 'VALID' "
                "  AND NVL(leaf_blocks, 0) > 1000 "
                "  AND NVL(num_rows, 0) > 0 "
                "ORDER BY fragmentation_pct DESC, blevel DESC "
                "FETCH FIRST 25 ROWS ONLY"
            ),
            # ── tables ──
            "tbl_stats": (
                "SELECT owner, table_name, num_rows FROM dba_tables "
                "WHERE num_rows IS NOT NULL ORDER BY num_rows DESC FETCH FIRST 20 ROWS ONLY"
            ),
            "tbl_large": (
                "SELECT owner, segment_name AS tablename, bytes FROM dba_segments "
                "WHERE segment_type='TABLE' ORDER BY bytes DESC FETCH FIRST 20 ROWS ONLY"
            ),
            "tbl_ts": (
                "SELECT df.tablespace_name, "
                "       ROUND(df.total_mb, 2) AS total_mb, "
                "       ROUND(NVL(fs.free_mb, 0), 2) AS free_mb, "
                "       ROUND(df.total_mb - NVL(fs.free_mb, 0), 2) AS used_mb, "
                "       ROUND(CASE WHEN df.total_mb = 0 THEN 0 "
                "                  ELSE ((df.total_mb - NVL(fs.free_mb, 0)) / df.total_mb) * 100 END, 2) AS used_pct "
                "FROM (SELECT tablespace_name, SUM(bytes)/1024/1024 AS total_mb "
                "      FROM dba_data_files GROUP BY tablespace_name) df "
                "LEFT JOIN (SELECT tablespace_name, SUM(bytes)/1024/1024 AS free_mb "
                "           FROM dba_free_space GROUP BY tablespace_name) fs "
                "  ON df.tablespace_name = fs.tablespace_name "
                "ORDER BY used_pct DESC"
            ),
            "tbl_temp": (
                "SELECT tf.tablespace_name, "
                "       ROUND(SUM(tf.bytes)/1024/1024, 2) AS total_mb, "
                "       ROUND(SUM(NVL(th.bytes_free, 0))/1024/1024, 2) AS free_mb, "
                "       ROUND((SUM(tf.bytes) - SUM(NVL(th.bytes_free, 0)))/1024/1024, 2) AS used_mb, "
                "       ROUND(CASE WHEN SUM(tf.bytes) = 0 THEN 0 "
                "                  ELSE ((SUM(tf.bytes) - SUM(NVL(th.bytes_free, 0))) / SUM(tf.bytes)) * 100 END, 2) AS used_pct "
                "FROM dba_temp_files tf "
                "LEFT JOIN v$temp_space_header th "
                "  ON tf.tablespace_name = th.tablespace_name "
                "GROUP BY tf.tablespace_name "
                "ORDER BY used_pct DESC"
            ),
            # ── locks ──
            "lock_waits": (
                "SELECT sid, serial#, event, wait_class FROM v$session "
                "WHERE state='ACTIVE' AND wait_class != 'Idle'"
            ),
            "lock_stats": (
                "SELECT s.sid AS session_id, s.username AS oracle_username, "
                "l.type AS lock_type, l.lmode AS locked_mode, l.request, l.id1, l.id2 "
                "FROM v$lock l JOIN v$session s ON l.sid = s.sid "
                "WHERE l.lmode > 0 AND s.username IS NOT NULL "
                "FETCH FIRST 20 ROWS ONLY"
            ),
            # ── replication ──
            "repl": (
                "SELECT name, value, unit, time_computed "
                "FROM v$dataguard_stats WHERE rownum <= 20"
            ),
        }

        # Execute all queries in a single JVM session
        batch = self.conn.execute_batch_queries_dict(queries)
        _batch_ms = (_time.monotonic() - _t0) * 1000
        logger.info("Oracle batched queries executed in %.0f ms (1 JVM session, %d queries)",
                     _batch_ms, len(queries))

        # ── Assemble results into the same shape as sequential collection ──
        # general
        version = "Oracle"
        ver_rows = batch.get("gen_version", [])
        if ver_rows:
            version = str(ver_rows[0].get("BANNER") or ver_rows[0].get("banner") or "Oracle")

        uptime = "Unknown"
        up_rows = batch.get("gen_uptime", [])
        if up_rows:
            try:
                uptime_sec = _safe_float(up_rows[0].get('UPTIME_SECONDS') or up_rows[0].get('uptime_seconds') or 0, 0.0)
                uptime = f"{uptime_sec:.0f} seconds"
            except Exception:
                pass

        size_human, size_bytes = "Unknown", 0
        sz_rows = batch.get("gen_dbsize", [])
        if sz_rows:
            try:
                total = _safe_int(sz_rows[0].get("TOTAL_BYTES") or sz_rows[0].get("total_bytes") or 0, 0)
                size_bytes = total
                if total >= 1024 ** 3:
                    size_human = f"{total / (1024 ** 3):.2f} GB"
                elif total >= 1024 ** 2:
                    size_human = f"{total / (1024 ** 2):.2f} MB"
                else:
                    size_human = f"{total} B"
            except Exception:
                pass

        metrics: Dict[str, Any] = {}
        metrics['general'] = {
            'version': version,
            'database': {'size_human': size_human, 'size_bytes': size_bytes},
            'uptime': uptime,
        }

        # connections
        connections = batch.get("conn_by_status", [])
        active_row = batch.get("conn_active", [])
        active_connections = _safe_int(
            active_row[0].get('ACTIVE_CONNECTIONS') or active_row[0].get('active_connections') or 0,
            0,
        ) if active_row else 0
        max_row = batch.get("conn_max", [])
        max_connections = _safe_int(
            max_row[0].get('VALUE') or max_row[0].get('value') or 0,
            0,
        ) if max_row else 0
        connection_usage = round((active_connections / max_connections * 100) if max_connections > 0 else 0, 2)
        metrics['connections'] = {
            'connections': connections,
            'max_connections': max_connections,
            'active_connections': active_connections,
            'connection_usage_percent': connection_usage,
        }

        # cache
        overall_hit_ratio = 0
        cache_rows = batch.get("cache_hit", [])
        if cache_rows:
            try:
                val = cache_rows[0].get('HIT_RATIO') or cache_rows[0].get('hit_ratio')
                overall_hit_ratio = float(val or 0)
            except Exception:
                pass
        metrics['cache'] = {
            'table_cache_stats': [],
            'overall_hit_ratio': overall_hit_ratio,
        }

        # queries — apply SYS/internal filter to slow queries
        from api import _filter_oracle_sys_sql as _filt_batch
        slow_queries = _filt_batch(batch.get("query_slow", []))
        total_row = batch.get("query_total", [])
        total_queries = _safe_int(
            total_row[0].get('TOTAL_QUERIES') or total_row[0].get('total_queries') or 0,
            0,
        ) if total_row else 0
        metrics['queries'] = {
            'slow_queries': slow_queries,
            'total_queries': total_queries,
        }

        # indexes — try primary result; if empty, fallback handled by sequential method
        unused_indexes = batch.get("idx_unused", [])
        metrics['indexes'] = {
            'unused_indexes': unused_indexes,
            'large_indexes': batch.get("idx_large", []),
            'fragmented_indexes': batch.get("idx_frag", []),
            'fragmented_index_count': len(batch.get("idx_frag", [])),
        }

        # tables
        metrics['tables'] = {
            'table_stats': batch.get("tbl_stats", []),
            'large_tables': batch.get("tbl_large", []),
            'tablespace_usage': batch.get("tbl_ts", []),
            'temp_tablespace_usage': batch.get("tbl_temp", []),
        }

        # locks
        metrics['locks'] = {
            'waiting_locks': batch.get("lock_waits", []),
            'lock_statistics': batch.get("lock_stats", []),
        }

        # replication
        metrics['replication'] = {
            'replication_slots': batch.get("repl", []),
        }

        # Store batch for extracting live metrics without a second JVM call
        self._last_oracle_batch = batch

        return metrics

    def extract_oracle_live_metrics_from_batch(self) -> Dict[str, Any]:
        """Extract a live-metrics dict (same shape as _collect_oracle_live_metrics)
        from the already-executed batch, avoiding a second JVM session.

        Returns an empty dict if no batch data is available.
        """
        batch = getattr(self, '_last_oracle_batch', None)
        if not batch:
            return {}

        result: Dict[str, Any] = {}

        def _scalar(value: Any) -> Any:
            if isinstance(value, (list, tuple)):
                if not value:
                    return 0
                return _scalar(value[0])
            return value

        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                v = _scalar(value)
                if v in (None, ''):
                    return default
                return int(v)
            except (TypeError, ValueError):
                return default

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                v = _scalar(value)
                if v in (None, ''):
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        # db identity
        rows = batch.get("live_db_identity", [])
        logger.info("[extract_live] live_db_identity rows=%d, data=%s", len(rows), rows[:1] if rows else '[]')
        if rows:
            result['db_name'] = str(rows[0].get('DB_NAME') or rows[0].get('db_name') or '')
            result['dbid'] = rows[0].get('DBID') or rows[0].get('dbid')
            result['open_mode'] = str(rows[0].get('OPEN_MODE') or rows[0].get('open_mode') or '')

        # Fallback if v$database was inaccessible (privilege issue)
        if not result.get('db_name'):
            fb_rows = batch.get("live_db_identity_fallback", [])
            logger.info("[extract_live] live_db_identity_fallback rows=%d, data=%s", len(fb_rows), fb_rows[:1] if fb_rows else '[]')
            if fb_rows:
                result['db_name'] = str(fb_rows[0].get('DB_NAME') or fb_rows[0].get('db_name') or '')
                result['open_mode'] = result.get('open_mode') or str(fb_rows[0].get('DB_UNIQUE_NAME') or fb_rows[0].get('db_unique_name') or '')

        # Second fallback: derive from v$instance status
        if not result.get('open_mode') or result.get('open_mode') == 'n/a':
            inst_rows = batch.get("live_instance_status", [])
            if inst_rows:
                result['open_mode'] = str(inst_rows[0].get('STATUS') or inst_rows[0].get('status') or '')

        # sessions
        active_rows = batch.get("conn_active", [])
        result['active_sessions'] = _safe_int(
            (active_rows[0].get('ACTIVE_CONNECTIONS') or active_rows[0].get('active_connections') or 0) if active_rows else 0,
            0,
        )

        total_rows = batch.get("live_total_sess", [])
        result['total_sessions'] = _safe_int(
            (total_rows[0].get('TOTAL_SESSIONS') or total_rows[0].get('total_sessions') or 0) if total_rows else 0,
            0,
        )

        max_rows = batch.get("live_max_proc", [])
        result['max_processes'] = _safe_int(
            (max_rows[0].get('VALUE') or max_rows[0].get('value') or 0) if max_rows else 0,
            0,
        )

        # wait events
        result['wait_events'] = batch.get("live_wait_events", [])

        # top SQL (reuse from main batch) — apply SYS/internal filter
        from api import _filter_oracle_sys_sql
        result['top_sql'] = _filter_oracle_sys_sql(batch.get("query_slow", []))

        # SGA / PGA
        result['sga'] = batch.get("live_sga", [])
        result['pga'] = batch.get("live_pga", [])

        # cache hit
        cache_rows = batch.get("cache_hit", [])
        if cache_rows:
            result['buffer_cache_hit_pct'] = _safe_float(cache_rows[0].get('HIT_RATIO') or cache_rows[0].get('hit_ratio') or 0, 0.0)
        else:
            result['buffer_cache_hit_pct'] = 0

        # redo
        redo_rows = batch.get("live_redo_wait", [])
        result['redo_space_requests'] = _safe_int(
            (redo_rows[0].get('REDO_SPACE_REQUESTS') or redo_rows[0].get('redo_space_requests') or 0) if redo_rows else 0,
            0,
        )

        # lib cache
        lib_rows = batch.get("live_lib_cache", [])
        result['lib_cache_hit_pct'] = _safe_float(
            (lib_rows[0].get('LIB_CACHE_HIT_PCT') or lib_rows[0].get('lib_cache_hit_pct') or 0) if lib_rows else 0,
            0.0,
        )

        # db cache size
        dbc_rows = batch.get("live_db_cache_sz", [])
        result['db_cache_size'] = str(dbc_rows[0].get('DB_CACHE_SIZE') or dbc_rows[0].get('db_cache_size') or '0') if dbc_rows else '0'

        # OS stats
        os_rows = batch.get("live_os_stats", [])
        logger.info("[extract_live] live_os_stats rows=%d, live_db_cache_sz rows=%d", len(os_rows), len(dbc_rows))
        result['os_stats'] = {}
        for r in os_rows:
            sn = str(r.get('STAT_NAME') or r.get('stat_name') or '')
            sv = r.get('VALUE') or r.get('value') or 0
            result['os_stats'][sn] = sv

        # unusable indexes — health only needs the count, not full detail rows
        uix_rows = batch.get("live_unusable_cnt", [])
        result['unusable_indexes'] = _safe_int(
            (uix_rows[0].get('UNUSABLE_INDEXES') or uix_rows[0].get('unusable_indexes') or 0) if uix_rows else 0,
            0,
        )
        result['unusable_index_details'] = batch.get("live_unusable_det", [])

        # invalid objects
        inv_rows = batch.get("live_invalid_cnt", [])
        result['invalid_objects'] = _safe_int(
            (inv_rows[0].get('INVALID_OBJECTS') or inv_rows[0].get('invalid_objects') or 0) if inv_rows else 0,
            0,
        )

        # DB links
        all_links = batch.get("live_dblink_all", [])
        status_links = batch.get("live_dblink_status", [])
        # Prefer the status query (has VALID column on 12c+); fall back to plain list
        link_rows = status_links if status_links else all_links
        result['db_links'] = link_rows
        faulty_links = []
        for lk in link_rows:
            valid_flag = str(lk.get('VALID') or lk.get('valid') or 'YES').upper()
            if valid_flag != 'YES':
                faulty_links.append(lk)
        result['faulty_db_links'] = faulty_links
        result['faulty_db_link_count'] = len(faulty_links)
        result['total_db_link_count'] = len(link_rows)

        return result

    def probe_oracle_db_links(
        self,
        db_links: List[Dict[str, Any]],
        timeout_seconds: int = 30,
    ) -> List[Dict[str, Any]]:
        """Probe Oracle DB links using a single PL/SQL block for efficiency.

        Instead of executing N separate ``SELECT 1 FROM DUAL@link`` queries
        (each potentially hanging for 30s on unreachable hosts), this uses a
        single PL/SQL anonymous block that iterates all links with per-link
        exception handling.  The result is returned as a single query output.

        Args:
            db_links: List of link dicts (from ``dba_db_links``).  Each must
                      contain a ``DB_LINK`` or ``db_link`` key.
            timeout_seconds: Maximum wall-clock seconds for the whole probe.
                             Default 30s.

        Returns:
            A *new* list of link dicts, each enriched with ``probe_status``
            (REACHABLE | UNREACHABLE | TIMEOUT | UNKNOWN).
        """
        if not db_links:
            return db_links

        # Build sanitized link name list
        link_name_map: Dict[str, Dict[str, Any]] = {}  # safe_name -> original dict
        for lk in db_links:
            link_name = str(lk.get('DB_LINK') or lk.get('db_link') or '').strip()
            if not link_name:
                continue
            # Sanitise: only allow alphanumeric, underscore, dot, @
            safe_name = ''.join(c for c in link_name if c.isalnum() or c in '_.@')
            if safe_name:
                link_name_map[safe_name] = lk

        if not link_name_map:
            return db_links

        # ── Strategy: Single PL/SQL block with per-link EXECUTE IMMEDIATE ──
        # This creates a temp global temporary table (or uses DBMS_OUTPUT)
        # to return results. We use DBMS_OUTPUT for simplicity.
        plsql_lines = []
        for safe_name in link_name_map:
            # Each link gets an EXECUTE IMMEDIATE wrapped in BEGIN..EXCEPTION
            plsql_lines.append(
                f"  BEGIN\n"
                f"    EXECUTE IMMEDIATE 'SELECT 1 FROM DUAL@{safe_name}' INTO v_dummy;\n"
                f"    DBMS_OUTPUT.PUT_LINE('{safe_name}=REACHABLE');\n"
                f"  EXCEPTION WHEN OTHERS THEN\n"
                f"    DBMS_OUTPUT.PUT_LINE('{safe_name}=UNREACHABLE');\n"
                f"  END;"
            )

        plsql_block = (
            "DECLARE\n"
            "  v_dummy NUMBER;\n"
            "BEGIN\n"
            + "\n".join(plsql_lines) + "\n"
            "END;"
        )

        # ── Try native PL/SQL execution first (oracledb) ──
        probe_results: Dict[str, str] = {}  # safe_name -> status
        _t0 = time.monotonic()

        if hasattr(self.conn, 'connection') and self.conn.connection is not None:
            # Native oracledb connection — use cursor with reduced call_timeout
            try:
                conn_obj = self.conn.connection
                # Temporarily reduce call_timeout for the probe (5s per link, min 15s)
                old_timeout = getattr(conn_obj, 'call_timeout', 30000)
                per_link_ms = 5000  # 5 seconds per link max
                total_ms = max(15000, min(len(link_name_map) * per_link_ms, timeout_seconds * 1000))
                try:
                    conn_obj.call_timeout = total_ms
                except Exception:
                    pass

                cursor = conn_obj.cursor()
                try:
                    # Enable DBMS_OUTPUT
                    cursor.callproc("DBMS_OUTPUT.ENABLE", [1000000])
                    cursor.execute(plsql_block)
                    # Fetch DBMS_OUTPUT lines
                    status_var = cursor.var(int)
                    line_var = cursor.var(str)
                    while True:
                        cursor.callproc("DBMS_OUTPUT.GET_LINE", [line_var, status_var])
                        if status_var.getvalue() != 0:
                            break
                        line = str(line_var.getvalue() or '')
                        if '=' in line:
                            parts = line.split('=', 1)
                            probe_results[parts[0]] = parts[1]
                finally:
                    cursor.close()
                    # Restore original timeout
                    try:
                        conn_obj.call_timeout = old_timeout
                    except Exception:
                        pass

                _ms = (time.monotonic() - _t0) * 1000
                logger.info("DB link probe (native PL/SQL): %d links in %.0f ms, %d reachable",
                            len(link_name_map), _ms,
                            sum(1 for v in probe_results.values() if v == 'REACHABLE'))

            except Exception as ex:
                logger.warning("DB link probe (native) failed: %s — falling back to batch", ex)
                probe_results = {}

        # ── Fallback: MCP/SQLcl batch approach ──
        if not probe_results and hasattr(self.conn, 'execute_batch_queries_dict'):
            try:
                # For MCP, use serveroutput ON approach in a script
                # Build individual SELECT queries (simpler for SQLcl spool parsing)
                probe_queries: Dict[str, str] = {}
                label_to_name: Dict[str, str] = {}
                for safe_name in link_name_map:
                    label = f"probe_{safe_name.replace('.', '_').replace('@', '_')}"
                    probe_queries[label] = f"SELECT 1 AS alive FROM DUAL@{safe_name}"
                    label_to_name[label] = safe_name

                original_timeout = getattr(self.conn, 'timeout_seconds', 90)
                try:
                    self.conn.timeout_seconds = max(15, timeout_seconds)
                    batch_results = self.conn.execute_batch_queries_dict(probe_queries)
                finally:
                    self.conn.timeout_seconds = original_timeout

                for label, safe_name in label_to_name.items():
                    rows = batch_results.get(label, [])
                    probe_results[safe_name] = 'REACHABLE' if rows else 'UNREACHABLE'

                _ms = (time.monotonic() - _t0) * 1000
                logger.info("DB link probe (MCP batch): %d links in %.0f ms, %d reachable",
                            len(link_name_map), _ms,
                            sum(1 for v in probe_results.values() if v == 'REACHABLE'))

            except Exception as ex:
                logger.warning("DB link probe batch failed (timeout or error): %s", ex)
                # Mark all as TIMEOUT
                for safe_name in link_name_map:
                    probe_results[safe_name] = 'TIMEOUT'

        # ── Build enriched result list ──
        probed: List[Dict[str, Any]] = []
        probed_ids: set = set()
        for safe_name, lk in link_name_map.items():
            enriched = dict(lk)
            enriched['probe_status'] = probe_results.get(safe_name, 'UNKNOWN')
            probed.append(enriched)
            probed_ids.add(id(lk))

        # Include any links that weren't probed (e.g. empty/invalid name)
        for lk in db_links:
            if id(lk) not in probed_ids:
                enriched = dict(lk)
                enriched['probe_status'] = 'UNKNOWN'
                probed.append(enriched)

        return probed

    def _collect_oracle_alternate_plans(
        self,
        sql_id: str,
        current_plan_hash: Optional[int],
        force_signature: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Collect alternate Oracle plans for SQL with same force-matching signature."""
        if not sql_id:
            return []

        fms = force_signature
        if fms is None:
            try:
                sig_rows = self.conn.execute_query_dict(
                    "SELECT force_matching_signature FROM v$sql "
                    f"WHERE sql_id = '{sql_id}' AND rownum <= 1"
                )
                if sig_rows:
                    raw = sig_rows[0].get('FORCE_MATCHING_SIGNATURE') or sig_rows[0].get('force_matching_signature')
                    fms = int(raw) if raw is not None else None
            except Exception:
                fms = None

        if not fms:
            return []

        # --- Source 1: v$sql (cursor cache) ---
        try:
            rows = self.conn.execute_query_dict(
                "SELECT sql_id, plan_hash_value, executions, "
                "       ROUND(elapsed_time / NULLIF(executions, 0) / 1e6, 4) AS avg_elapsed_sec, "
                "       ROUND(cpu_time / NULLIF(executions, 0) / 1e6, 4) AS avg_cpu_sec, "
                "       last_active_time "
                "FROM ( "
                "  SELECT sql_id, plan_hash_value, executions, elapsed_time, cpu_time, last_active_time "
                "  FROM v$sql "
                f"  WHERE force_matching_signature = {int(fms)} "
                "    AND plan_hash_value IS NOT NULL "
                "  ORDER BY elapsed_time / NULLIF(executions, 0) "
                ") WHERE rownum <= 10"
            )
        except Exception:
            rows = []

        plans: List[Dict[str, Any]] = []
        for r in rows or []:
            phv_raw = r.get('PLAN_HASH_VALUE') or r.get('plan_hash_value')
            try:
                phv = int(phv_raw)
            except Exception:
                continue
            plans.append({
                'sql_id': str(r.get('SQL_ID') or r.get('sql_id') or sql_id),
                'plan_hash_value': phv,
                'executions': int(r.get('EXECUTIONS') or r.get('executions') or 0),
                'avg_elapsed_sec': float(r.get('AVG_ELAPSED_SEC') or r.get('avg_elapsed_sec') or 0),
                'avg_cpu_sec': float(r.get('AVG_CPU_SEC') or r.get('avg_cpu_sec') or 0),
                'last_active_time': str(r.get('LAST_ACTIVE_TIME') or r.get('last_active_time') or ''),
                'is_current_plan': bool(current_plan_hash is not None and phv == current_plan_hash),
                'source': 'cursor_cache',
            })

        # --- Source 2: AWR (awr_root_sqlstat) — catches aged-out plans ---
        # First with the snapshot join, then without it as a permissive fallback.
        seen_awr_phvs = set()
        for awr_stat_view, awr_snap_view in [
            ('awr_root_sqlstat', 'awr_root_snapshot'),
            ('awr_root_sqlstat', None),
        ]:
            try:
                if awr_snap_view:
                    awr_rows = self.conn.execute_query_dict(
                        f"SELECT hs.sql_id, hs.plan_hash_value, "
                        f"       SUM(hs.executions_delta) AS executions, "
                        f"       ROUND(SUM(hs.elapsed_time_delta)/NULLIF(SUM(hs.executions_delta),0)/1e6, 4) AS avg_elapsed_sec, "
                        f"       ROUND(SUM(hs.cpu_time_delta)/NULLIF(SUM(hs.executions_delta),0)/1e6, 4) AS avg_cpu_sec, "
                        f"       MAX(sn.end_interval_time) AS last_active_time "
                        f"FROM {awr_stat_view} hs "
                        f"JOIN {awr_snap_view} sn "
                        f"  ON sn.dbid = hs.dbid AND sn.instance_number = hs.instance_number AND sn.snap_id = hs.snap_id "
                        f"WHERE hs.sql_id = '{sql_id}' "
                        f"  AND hs.plan_hash_value IS NOT NULL AND hs.plan_hash_value != 0 "
                        f"  AND sn.begin_interval_time >= SYSDATE - 30 "
                        f"GROUP BY hs.sql_id, hs.plan_hash_value "
                        f"ORDER BY SUM(hs.elapsed_time_delta) DESC "
                        f"FETCH FIRST 10 ROWS ONLY"
                    ) or []
                else:
                    awr_rows = self.conn.execute_query_dict(
                        f"SELECT sql_id, plan_hash_value, "
                        f"       SUM(executions_delta) AS executions, "
                        f"       ROUND(SUM(elapsed_time_delta)/NULLIF(SUM(executions_delta),0)/1e6, 4) AS avg_elapsed_sec, "
                        f"       ROUND(SUM(cpu_time_delta)/NULLIF(SUM(executions_delta),0)/1e6, 4) AS avg_cpu_sec, "
                        f"       NULL AS last_active_time "
                        f"FROM {awr_stat_view} "
                        f"WHERE sql_id = '{sql_id}' "
                        f"  AND plan_hash_value IS NOT NULL AND plan_hash_value != 0 "
                        f"GROUP BY sql_id, plan_hash_value "
                        f"ORDER BY SUM(elapsed_time_delta) DESC "
                        f"FETCH FIRST 10 ROWS ONLY"
                    ) or []
            except Exception:
                awr_rows = []
                continue
            # Add rows for phvs we haven't seen yet
            for r in awr_rows:
                phv_raw = r.get('PLAN_HASH_VALUE') or r.get('plan_hash_value')
                try:
                    phv = int(phv_raw)
                except Exception:
                    continue
                if phv in seen_awr_phvs:
                    continue
                seen_awr_phvs.add(phv)
                plans.append({
                    'sql_id': str(r.get('SQL_ID') or r.get('sql_id') or sql_id),
                    'plan_hash_value': phv,
                    'executions': int(r.get('EXECUTIONS') or r.get('executions') or 0),
                    'avg_elapsed_sec': float(r.get('AVG_ELAPSED_SEC') or r.get('avg_elapsed_sec') or 0),
                    'avg_cpu_sec': float(r.get('AVG_CPU_SEC') or r.get('avg_cpu_sec') or 0),
                    'last_active_time': str(r.get('LAST_ACTIVE_TIME') or r.get('last_active_time') or ''),
                    'is_current_plan': bool(current_plan_hash is not None and phv == current_plan_hash),
                    'source': 'AWR',
                })

        # Dedup: prefer cursor_cache entry over AWR for same plan_hash_value
        dedup: Dict[int, Dict[str, Any]] = {}
        for p in plans:
            key = int(p['plan_hash_value'])
            if key not in dedup or p.get('source') == 'cursor_cache':
                dedup[key] = p
        return list(dedup.values())

    def _compare_oracle_plans(
        self,
        current_avg_sec: float,
        current_plan_hash: Optional[int],
        current_executions: int,
        alternate_plans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare current plan against alternates and produce confidence score.
        
        Returns a ranked summary of ALL plans plus a determination of whether
        any alternate plan is better than the current one.
        """
        current_avg = float(current_avg_sec or 0)

        # Build ranked list of ALL plans (not just alternates)
        ranked_plans = []
        for p in alternate_plans:
            avg_sec = float(p.get('avg_elapsed_sec') or 0)
            phv = p.get('plan_hash_value')
            is_cur = bool(current_plan_hash is not None and phv == current_plan_hash)
            ranked_plans.append({
                'plan_hash_value': phv,
                'avg_elapsed_sec': round(avg_sec, 4),
                'avg_cpu_sec': round(float(p.get('avg_cpu_sec') or 0), 4),
                'executions': int(p.get('executions') or 0),
                'source': p.get('source', ''),
                'last_active_time': p.get('last_active_time', ''),
                'first_load_time': p.get('first_load_time', ''),
                'is_current_plan': is_cur,
                'is_best': False,  # set below
                'rank': 0,         # set below
            })

        # Sort: plans with execution data first (by avg_elapsed_sec ASC),
        # then plans with no execution data last
        has_stats = [p for p in ranked_plans if p['avg_elapsed_sec'] > 0]
        no_stats = [p for p in ranked_plans if p['avg_elapsed_sec'] <= 0]
        has_stats.sort(key=lambda p: p['avg_elapsed_sec'])
        ranked_plans = has_stats + no_stats
        for i, p in enumerate(ranked_plans):
            p['rank'] = i + 1
        if ranked_plans:
            ranked_plans[0]['is_best'] = True

        others = [
            p for p in alternate_plans
            if p.get('plan_hash_value') != current_plan_hash and float(p.get('avg_elapsed_sec') or 0) > 0
        ]
        if not others:
            return {
                'status': 'single_plan_or_no_alternates',
                'better_plan_found': False,
                'current_plan_hash_value': current_plan_hash,
                'best_plan_hash_value': current_plan_hash,
                'confidence_pct': 0.0,
                'total_plans': len(alternate_plans),
                'total_alternate_plans': 0,
                'ranked_plans': ranked_plans,
            }

        best = min(others, key=lambda p: float(p.get('avg_elapsed_sec') or 0))
        best_avg = float(best.get('avg_elapsed_sec') or 0)
        improvement_pct = 0.0
        if current_avg > 0 and best_avg > 0:
            improvement_pct = ((current_avg - best_avg) / current_avg) * 100.0

        sample_factor = min(1.0, (float(current_executions or 0) + float(best.get('executions') or 0)) / 100.0)
        gap_factor = min(1.0, max(0.0, improvement_pct) / 50.0)
        diversity_factor = min(1.0, len(others) / 3.0)
        confidence_pct = round((0.55 * gap_factor + 0.35 * sample_factor + 0.10 * diversity_factor) * 100.0, 1)

        better = improvement_pct > 5.0
        # best_plan_hash_value = the absolute best plan (current or alternate)
        # best_alternate_hash_value = the best plan that is NOT current
        absolute_best_phv = best.get('plan_hash_value') if better else current_plan_hash
        return {
            'status': 'better_alternate_found' if better else 'current_plan_competitive',
            'better_plan_found': better,
            'current_plan_hash_value': current_plan_hash,
            'best_plan_hash_value': absolute_best_phv,
            'best_alternate_hash_value': best.get('plan_hash_value'),
            'current_avg_elapsed_sec': round(current_avg, 4),
            'best_avg_elapsed_sec': round(best_avg, 4),
            'estimated_improvement_pct': round(improvement_pct, 2),
            'confidence_pct': confidence_pct,
            'total_plans': len(alternate_plans),
            'total_alternate_plans': len(others),
            'ranked_plans': ranked_plans,
        }

    def _collect_connection_metrics(self) -> Dict[str, Any]:
        """Collect connection and backend metrics."""
        query = """
        SELECT 
            datname as database,
            usename as user,
            application_name,
            state,
            count(*) as connection_count
        FROM pg_stat_activity 
        WHERE usename IS NOT NULL
        GROUP BY datname, usename, application_name, state
        ORDER BY connection_count DESC;
        """
        connections = self.conn.execute_query_dict(query)
        
        # Read actual max_connections from pg_settings
        max_connections = 100
        try:
            mc_row = self.conn.execute_query_dict(
                "SELECT setting::int AS max_connections FROM pg_settings WHERE name = 'max_connections'"
            )
            if mc_row:
                max_connections = int(mc_row[0].get('max_connections') or 100)
        except Exception:
            pass

        # Count active connections
        active_connections = 0
        try:
            active_query = """
            SELECT COUNT(*) as active_connections
            FROM pg_stat_activity
            WHERE state != 'idle' AND state IS NOT NULL;
            """
            active_conn = self.conn.execute_query_dict(active_query)
            if active_conn:
                active_connections = int(active_conn[0]['active_connections'] or 0)
        except Exception:
            active_connections = 0

        # Total open connections (not just active)
        total_connections = 0
        try:
            total_row = self.conn.execute_query_dict(
                "SELECT COUNT(*) AS total FROM pg_stat_activity"
            )
            if total_row:
                total_connections = int(total_row[0].get('total') or 0)
        except Exception:
            total_connections = active_connections
        
        connection_usage = round(
            (total_connections / max_connections * 100) if max_connections > 0 else 0, 2
        )
        
        return {
            'connections': connections,
            'max_connections': max_connections,
            'active_connections': active_connections,
            'total_connections': total_connections,
            'connection_usage_percent': connection_usage
        }
    
    def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect cache and buffer hit ratio metrics."""
        query = """
        SELECT 
            schemaname,
            relname as tablename,
            heap_blks_read,
            heap_blks_hit,
            CASE 
                WHEN (heap_blks_read + heap_blks_hit) > 0 
                THEN round(100.0 * heap_blks_hit / (heap_blks_read + heap_blks_hit), 2)
                ELSE 0
            END as hit_ratio
        FROM pg_statio_user_tables
        ORDER BY heap_blks_read DESC
        LIMIT 20;
        """
        table_cache = self.conn.execute_query_dict(query)
        
        # Global cache stats
        global_cache_query = """
        SELECT 
            sum(heap_blks_read) as total_reads,
            sum(heap_blks_hit) as total_hits,
            CASE 
                WHEN (sum(heap_blks_read) + sum(heap_blks_hit)) > 0
                THEN round(100.0 * sum(heap_blks_hit) / (sum(heap_blks_read) + sum(heap_blks_hit)), 2)
                ELSE 0
            END as overall_hit_ratio
        FROM pg_statio_user_tables;
        """
        global_cache = self.conn.execute_query_dict(global_cache_query)
        overall_hit_ratio = 0
        if global_cache and len(global_cache) > 0:
            overall_hit_ratio = global_cache[0]['overall_hit_ratio'] if global_cache[0]['overall_hit_ratio'] is not None else 0
        
        return {
            'table_cache_stats': table_cache,
            'overall_hit_ratio': overall_hit_ratio
        }
    
    def _collect_query_metrics(self) -> Dict[str, Any]:
        """Collect query performance metrics from pg_stat_statements."""
        try:
            query = """
            SELECT 
                query,
                calls,
                total_exec_time as total_time,
                mean_exec_time as mean_time,
                max_exec_time as max_time,
                stddev_exec_time as stddev_time,
                rows
            FROM pg_stat_statements
            WHERE query NOT LIKE '%%pg_stat_statements%%'
            ORDER BY total_exec_time DESC
            LIMIT 20;
            """
            slow_queries = self.conn.execute_query_dict(query)
        except Exception as e:
            logger.warning(f"Could not collect pg_stat_statements data (may use different column names): {e}")
            slow_queries = []
        
        # Get query count
        try:
            count_query = """
            SELECT COUNT(*) as total_queries FROM pg_stat_statements;
            """
            query_count = self.conn.execute_query_dict(count_query)
            total_queries = 0
            if query_count and len(query_count) > 0:
                total_queries = query_count[0]['total_queries']
        except Exception as e:
            logger.warning(f"Could not get query count: {e}")
            total_queries = 0
        
        return {
            'slow_queries': slow_queries,
            'total_queries': total_queries
        }
    
    def _collect_index_metrics(self) -> Dict[str, Any]:
        """Collect index usage and bloat metrics."""
        query = """
        SELECT 
            schemaname,
            relname as table_name,
            indexrelname as indexname,
            idx_scan as index_scans,
            idx_tup_read as tuples_read,
            idx_tup_fetch as tuples_fetched
        FROM pg_stat_user_indexes
        WHERE idx_scan = 0
        ORDER BY idx_scan DESC;
        """
        unused_indexes = self.conn.execute_query_dict(query)
        
        # Index size
        size_query = """
        SELECT 
            schemaname,
            relname as table_name,
            indexrelname as indexname,
            pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
            pg_relation_size(indexrelid)::bigint as size_bytes
        FROM pg_stat_user_indexes
        ORDER BY pg_relation_size(indexrelid) DESC
        LIMIT 20;
        """
        large_indexes = self.conn.execute_query_dict(size_query)
        
        return {
            'unused_indexes': unused_indexes,
            'large_indexes': large_indexes
        }
    
    def _collect_table_metrics(self) -> Dict[str, Any]:
        """Collect table bloat and access metrics."""
        query = """
        SELECT 
            schemaname,
            relname as tablename,
            seq_scan,
            seq_tup_read,
            idx_scan,
            idx_tup_fetch,
            n_tup_ins,
            n_tup_upd,
            n_tup_del,
            n_live_tup,
            n_dead_tup,
            last_vacuum,
            last_autovacuum
        FROM pg_stat_user_tables
        WHERE n_live_tup > 1000
        ORDER BY n_dead_tup DESC
        LIMIT 20;
        """
        table_stats = self.conn.execute_query_dict(query)
        
        # Table size
        size_query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
            pg_total_relation_size(schemaname||'.'||tablename)::bigint as size_bytes
        FROM pg_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        LIMIT 20;
        """
        large_tables = self.conn.execute_query_dict(size_query)
        
        return {
            'table_stats': table_stats,
            'large_tables': large_tables
        }
    
    def _collect_lock_metrics(self) -> Dict[str, Any]:
        """Collect lock contention metrics."""
        query = """
        SELECT 
            pid,
            usename,
            application_name,
            state,
            wait_event_type,
            wait_event,
            query_start,
            state_change
        FROM pg_stat_activity
        WHERE wait_event_type IS NOT NULL
        ORDER BY query_start;
        """
        locks = self.conn.execute_query_dict(query)
        
        # Lock statistics
        lock_stats_query = """
        SELECT 
            locktype,
            mode,
            COUNT(*) as lock_count
        FROM pg_locks
        GROUP BY locktype, mode
        ORDER BY lock_count DESC;
        """
        lock_stats = self.conn.execute_query_dict(lock_stats_query)
        
        return {
            'waiting_locks': locks,
            'lock_statistics': lock_stats
        }
    
    def _collect_replication_metrics(self) -> Dict[str, Any]:
        """Collect replication metrics if available."""
        query = """
        SELECT 
            pid,
            usename,
            application_name,
            client_addr,
            client_hostname,
            state,
            write_lsn,
            flush_lsn,
            replay_lsn,
            sync_state
        FROM pg_stat_replication;
        """
        replication = self.conn.execute_query_dict(query)
        
        return {
            'replication_slots': replication
        }


class MetricsAnalyzer:
    """Analyzes collected metrics against thresholds."""
    
    # Performance thresholds
    THRESHOLDS = {
        'cache_hit_ratio': 99.0,  # Minimum acceptable hit ratio %
        'max_connection_usage': 80.0,  # % of max connections
        'max_dead_tuples': 10000,  # Dead tuples per table
        'slow_query_threshold_ms': 5000,  # Query execution time
        'index_scan_threshold': 0,  # Unused index detection
        'lock_wait_threshold': 1000,  # Lock wait time ms
    }
    
    def __init__(self, metrics: Dict[str, Any]):
        """Initialize analyzer with metrics."""
        self.metrics = metrics
        self.issues = []
    
    def analyze(self) -> List[Dict[str, str]]:
        """
        Analyze metrics against thresholds.
        
        Returns:
            List of identified issues
        """
        self._check_cache_performance()
        self._check_connection_usage()
        self._check_dead_tuples()
        self._check_slow_queries()
        self._check_unused_indexes()
        self._check_locks()
        
        return self.issues
    
    def _check_cache_performance(self):
        """Check cache hit ratio against threshold."""
        hit_ratio = self.metrics.get('cache', {}).get('overall_hit_ratio', 100)
        if hit_ratio < self.THRESHOLDS['cache_hit_ratio']:
            self.issues.append({
                'severity': 'HIGH',
                'category': 'Cache Performance',
                'issue': f'Low cache hit ratio: {hit_ratio}%',
                'recommendation': 'Consider increasing shared_buffers or optimizing queries'
            })
    
    def _check_connection_usage(self):
        """Check connection pool usage."""
        usage = self.metrics.get('connections', {}).get('connection_usage_percent', 0)
        if usage > self.THRESHOLDS['max_connection_usage']:
            self.issues.append({
                'severity': 'HIGH',
                'category': 'Connections',
                'issue': f'High connection usage: {usage}%',
                'recommendation': 'Consider increasing max_connections or closing unused connections'
            })
    
    def _check_dead_tuples(self):
        """Check for excessive dead tuples."""
        table_stats = self.metrics.get('tables', {}).get('table_stats', [])
        for table in table_stats:
            if table.get('n_dead_tup', 0) > self.THRESHOLDS['max_dead_tuples']:
                self.issues.append({
                    'severity': 'MEDIUM',
                    'category': 'Table Maintenance',
                    'issue': f"Table {table['tablename']} has {table['n_dead_tup']} dead tuples",
                    'recommendation': 'Run VACUUM ANALYZE on this table'
                })
    
    def _check_slow_queries(self):
        """Check for slow queries."""
        slow_queries = self.metrics.get('queries', {}).get('slow_queries', [])
        for query in slow_queries[:5]:  # Top 5
            if query.get('mean_time', 0) > self.THRESHOLDS['slow_query_threshold_ms']:
                self.issues.append({
                    'severity': 'MEDIUM',
                    'category': 'Query Performance',
                    'issue': f"Slow query detected: {query.get('mean_time', 0)}ms avg time",
                    'recommendation': 'EXPLAIN ANALYZE the query and consider adding indexes'
                })
    
    def _check_unused_indexes(self):
        """Check for unused indexes."""
        unused = self.metrics.get('indexes', {}).get('unused_indexes', [])
        if len(unused) > 0:
            self.issues.append({
                'severity': 'LOW',
                'category': 'Index Maintenance',
                'issue': f'Found {len(unused)} unused indexes',
                'recommendation': 'Consider dropping unused indexes to save space'
            })
    
    def _check_locks(self):
        """Check for lock contention."""
        locks = self.metrics.get('locks', {}).get('waiting_locks', [])
        if len(locks) > 0:
            self.issues.append({
                'severity': 'MEDIUM',
                'category': 'Lock Contention',
                'issue': f'{len(locks)} queries waiting for locks',
                'recommendation': 'Investigate blocking queries and optimize transaction handling'
            })
