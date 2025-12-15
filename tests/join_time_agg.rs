//! # Distributed Join with Time-Based Aggregation Test
//!
//! This test demonstrates a distributed join followed by time-based aggregation.
//!
//! ## Scenario: Join + Time-Based Aggregation with Default Task Estimator
//! - Uses `FileScanConfigTaskEstimator` (default)
//! - Data: `testdata/join_test_hive/` with Hive-style partitioning (d_dkey=A/, f_dkey=A/, etc.)
//! - Workers: 2 workers, each processing 2 partitions (A,B on worker 0, C,D on worker 1)
//! - Query: Join dim and fact tables, then aggregate by f_dkey and timestamp with MAX(env) and MAX(value)
//! - Result: Multi-stage execution due to aggregation after join
//!
//! ## Key Takeaway
//! Even with optimal Hive-style partitioning enabling collocated joins, aggregation operations
//! after the join may require additional stages for proper data distribution.

#[cfg(all(feature = "integration", test))]
mod tests {
    use datafusion::arrow::array::Array;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::error::DataFusionError;
    use datafusion::execution::{SessionState, SessionStateBuilder};
    use datafusion::physical_plan::execute_stream;
    use datafusion::prelude::CsvReadOptions;
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::{DistributedSessionBuilderContext, display_plan_ascii};
    use futures::TryStreamExt;
    use std::error::Error;

    async fn build_default_state(
        ctx: DistributedSessionBuilderContext,
    ) -> Result<SessionState, DataFusionError> {
        Ok(SessionStateBuilder::new()
            .with_runtime_env(ctx.runtime_env)
            .with_default_features()
            .build())
    }

    #[tokio::test]
    async fn test_join_with_time_aggregation_and_default_task_estimator() -> Result<(), Box<dyn Error>> {
        // Start distributed context with 2 workers using DEFAULT task estimator
        let (ctx_distributed, _guard) = start_localhost_context(2, build_default_state).await;

        // Enable file partitioning preservation (same as test_join_with_default_task_estimator)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Set target_partitions to 4 to create 4 file groups (one per Hive partition: A, B, C, D)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register dimension table with Hive-style partitioning
        let dim_options = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options)
            .await?;

        // Register fact table with Hive-style partitioning
        // IMPORTANT: Declare sort order as (f_dkey, timestamp) following the pattern from
        // https://github.com/gene-bordegaray/datafusion/pull/3/files
        // 
        // This enables the hash superset optimization: when aggregating by (f_dkey, timestamp),
        // DataFusion recognizes that partitioning by f_dkey is sufficient (superset),
        // avoiding unnecessary repartitioning.
        let fact_options = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)])
            .file_sort_order(vec![ // NGA: This sort order definition seems not working
                vec![
                    datafusion::prelude::col("f_dkey").sort(true, true), // ASC NULLS FIRST
                    datafusion::prelude::col("timestamp").sort(true, true), // ASC NULLS FIRST
                ]
            ]);
        ctx_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options)
            .await?;

        // Create a join query with time-based aggregation
        let query = r#"
            SELECT  f_dkey, 
                    timestamp,
                    MAX(env) AS max_env,
                    MAX(value) AS max_value
            FROM
               (
                SELECT 
                    f.f_dkey,
                    d.env,
                    d.service,
                    d.host,
                    f.timestamp,
                    f.value
                FROM dim d
                INNER JOIN fact f ON d.d_dkey = f.f_dkey
               ) AS j
            GROUP BY f_dkey, timestamp
            ORDER BY f_dkey, timestamp
        "#;

        // Execute distributed query
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        println!("\n=== DISTRIBUTED PLAN ===");
        println!("{}", physical_distributed_str);

        // Execute and collect distributed results
        let batches_distributed = execute_stream(physical_distributed.clone(), ctx_distributed.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;
        let result_distributed = pretty_format_batches(&batches_distributed)?;

        println!("\n=== DISTRIBUTED RESULTS ===");
        println!("{}", result_distributed);

        // Create a non-distributed context for comparison
        use datafusion::prelude::SessionContext;
        let ctx_non_distributed = SessionContext::new();

        // Enable file partitioning preservation in non-distributed context too
        ctx_non_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Set target_partitions to 4 (same as distributed context)
        ctx_non_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register the same tables in non-distributed context
        let dim_options_nd = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_non_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options_nd)
            .await?;

        let fact_options_nd = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)])
            .file_sort_order(vec![
                vec![
                    datafusion::prelude::col("f_dkey").sort(true, true),
                    datafusion::prelude::col("timestamp").sort(true, true),
                ]
            ]);
        ctx_non_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options_nd)
            .await?;

        // Execute the same query in non-distributed context
        let df_non_distributed = ctx_non_distributed.sql(query).await?;
        let physical_non_distributed = df_non_distributed.clone().create_physical_plan().await?;
        let physical_non_distributed_str = display_plan_ascii(physical_non_distributed.as_ref(), false);

        println!("\n=== NON-DISTRIBUTED PLAN ===");
        println!("{}", physical_non_distributed_str);

        let batches_non_distributed = df_non_distributed.collect().await?;
        let result_non_distributed = pretty_format_batches(&batches_non_distributed)?;

        println!("\n=== NON-DISTRIBUTED RESULTS ===");
        println!("{}", result_non_distributed);

        // Compare results: both should have the same data
        let total_rows_distributed: usize = batches_distributed.iter().map(|b| b.num_rows()).sum();
        let total_rows_non_distributed: usize = batches_non_distributed.iter().map(|b| b.num_rows()).sum();

        assert_eq!(
            total_rows_distributed, total_rows_non_distributed,
            "Row count mismatch: distributed={}, non-distributed={}",
            total_rows_distributed, total_rows_non_distributed
        );

        // Compare the actual data by converting to strings and comparing
        // Since both are sorted by (f_dkey, timestamp), they should be identical
        assert_eq!(
            result_distributed.to_string(),
            result_non_distributed.to_string(),
            "Results differ between distributed and non-distributed execution"
        );

        println!("\n✓ Verified: Distributed and non-distributed results are identical ({} rows)", total_rows_distributed);

        // Expected distributed plan with join + time-based aggregation (OPTIMAL WITH HASH SUPERSET):
        // With the hash superset optimization from https://github.com/gene-bordegaray/datafusion/pull/3
        // This achieves a SINGLE-STAGE execution plan!
        // 
        // Stage 1 ONLY (Distributed):
        //   - Collocated join using Hive partitioning (NO shuffle!)
        //   - Single-pass aggregation (mode=SinglePartitioned) - NO repartitioning!
        //   - Sort for final ORDER BY
        // 
        // KEY OPTIMIZATION: Hash Superset Satisfies Partitioning
        // - The join is partitioned by d_dkey (via Hive partitioning)
        // - The aggregation needs (d_dkey, timestamp)
        // - Since d_dkey is a SUPERSET of the join partitioning, NO repartition is needed!
        // - DataFusion recognizes that data partitioned by d_dkey is sufficient for aggregating by (d_dkey, timestamp)
        // 
        // Benefits:
        // 1. ✅ Single stage instead of 2 stages
        // 2. ✅ NO RepartitionExec - eliminates network shuffle for aggregation
        // 3. ✅ NO NetworkShuffleExec between stages
        // 4. ✅ mode=SinglePartitioned instead of Partial+Final - single-pass aggregation
        // 5. ✅ ordering_mode=Sorted - leverages sorted fact table
        // 6. ✅ Minimal data movement - only final sorted results coalesced
        let expected_plan = r#"┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST]
│   [Stage 1] => NetworkCoalesceExec: output_partitions=4, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 1 ── Tasks: t0:[p0..p1] t1:[p2..p3] 
  │ SortExec: expr=[f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST], preserve_partitioning=[true]
  │   ProjectionExec: expr=[f_dkey@0 as f_dkey, timestamp@1 as timestamp, max(j.env)@2 as max_env, max(j.value)@3 as max_value]
  │     AggregateExec: mode=SinglePartitioned, gby=[f_dkey@0 as f_dkey, timestamp@2 as timestamp], aggr=[max(j.env), max(j.value)], ordering_mode=Sorted
  │       ProjectionExec: expr=[f_dkey@3 as f_dkey, env@0 as env, timestamp@1 as timestamp, value@2 as value]
  │         HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@1, f_dkey@2)], projection=[env@0, timestamp@2, value@3, f_dkey@4]
  │           PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │             DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, d_dkey], file_type=csv, has_header=true
  │           PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │             DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
  └──────────────────────────────────────────────────
"#;

        // Normalize paths for comparison
        let normalize_paths = |s: &str| -> String {
            s.lines()
                .map(|line| {
                    if line.contains("testdata/join_test_hive") {
                        let re = regex::Regex::new(r"[^,\s\[]*(/testdata/join_test_hive/[^,\s\]]+)").unwrap();
                        re.replace_all(line, "$1").to_string()
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        
        let normalized_expected = normalize_paths(&expected_plan);
        let normalized_actual = normalize_paths(&physical_distributed_str);
        
        assert_eq!(
            normalized_actual.trim(),
            normalized_expected.trim(),
            "\n\nActual plan does not match expected plan!\n\nExpected:\n{}\n\nActual:\n{}\n",
            normalized_expected,
            normalized_actual
        );

        // Expected non-distributed plan (with preserve_file_partitions and target_partitions=4)
        // With these settings, non-distributed also benefits from hash superset optimization!
        let expected_non_distributed_plan = r#"SortPreservingMergeExec: [f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST]
  SortExec: expr=[f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST], preserve_partitioning=[true]
    ProjectionExec: expr=[f_dkey@0 as f_dkey, timestamp@1 as timestamp, max(j.env)@2 as max_env, max(j.value)@3 as max_value]
      AggregateExec: mode=SinglePartitioned, gby=[f_dkey@0 as f_dkey, timestamp@2 as timestamp], aggr=[max(j.env), max(j.value)], ordering_mode=Sorted
        ProjectionExec: expr=[f_dkey@3 as f_dkey, env@0 as env, timestamp@1 as timestamp, value@2 as value]
          HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@1, f_dkey@2)], projection=[env@0, timestamp@2, value@3, f_dkey@4]
            DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, d_dkey], file_type=csv, has_header=true
            DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
"#;

        let normalized_expected_nd = normalize_paths(&expected_non_distributed_plan);
        let normalized_actual_nd = normalize_paths(&physical_non_distributed_str);
        
        assert_eq!(
            normalized_actual_nd.trim(),
            normalized_expected_nd.trim(),
            "\n\nActual non-distributed plan does not match expected!\n\nExpected:\n{}\n\nActual:\n{}\n",
            normalized_expected_nd,
            normalized_actual_nd
        );

        // Verify we got results
        let total_rows: usize = batches_distributed.iter().map(|b| b.num_rows()).sum();
        assert!(
            total_rows > 0,
            "Should have some aggregated rows, got {}",
            total_rows
        );

        // Verify that all f_dkey values are present (A, B, C, D)
        let mut f_dkey_values = std::collections::HashSet::new();
        for batch in &batches_distributed {
            let f_dkey_array = batch
                .column_by_name("f_dkey")
                .expect("f_dkey column should exist");
            
            let f_dkey_strings = datafusion::arrow::array::cast::as_string_array(f_dkey_array);
            for i in 0..f_dkey_strings.len() {
                f_dkey_values.insert(f_dkey_strings.value(i).to_string());
            }
        }

        // Should have all 4 partition keys
        assert!(f_dkey_values.contains("A"), "Should have partition A");
        assert!(f_dkey_values.contains("B"), "Should have partition B");
        assert!(f_dkey_values.contains("C"), "Should have partition C");
        assert!(f_dkey_values.contains("D"), "Should have partition D");

        println!("\n✓ Verified: All partitions (A, B, C, D) present in aggregated results");
        println!("✓ Total aggregated rows: {}", total_rows);
        println!("✓ Unique f_dkey values: {:?}", f_dkey_values);

        Ok(())
    }

    #[tokio::test]
    async fn test_join_with_time_bin_aggregation_and_default_task_estimator() -> Result<(), Box<dyn Error>> {
        // Start distributed context with 2 workers using DEFAULT task estimator
        let (ctx_distributed, _guard) = start_localhost_context(2, build_default_state).await;

        // Enable file partitioning preservation
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Set target_partitions to 4 to create 4 file groups (one per Hive partition: A, B, C, D)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register dimension table with Hive-style partitioning
        let dim_options = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options)
            .await?;

        // Register fact table with Hive-style partitioning and sort order
        // IMPORTANT: Declare sort order as (f_dkey, timestamp) for hash superset optimization
        let fact_options = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)])
            .file_sort_order(vec![
                vec![
                    datafusion::prelude::col("f_dkey").sort(true, true), // ASC NULLS FIRST
                    datafusion::prelude::col("timestamp").sort(true, true), // ASC NULLS FIRST
                ]
            ]);
        ctx_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options)
            .await?;

        // Create a join query with date_bin aggregation (30-second time windows)
        let query = r#"
            SELECT  f_dkey, 
                    date_bin(INTERVAL '30 seconds', timestamp) AS time_bin,
                    MAX(env) AS max_env,
                    MAX(value) AS max_value
            FROM
               (
                SELECT 
                    f.f_dkey,
                    d.env,
                    d.service,
                    d.host,
                    f.timestamp,
                    f.value
                FROM dim d
                INNER JOIN fact f ON d.d_dkey = f.f_dkey
               ) AS j
            GROUP BY f_dkey, time_bin
            ORDER BY f_dkey, time_bin
        "#;

        // Execute distributed query
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        println!("\n=== DISTRIBUTED PLAN (with date_bin) ===");
        println!("{}", physical_distributed_str);

        // Execute and collect distributed results
        let batches_distributed = execute_stream(physical_distributed.clone(), ctx_distributed.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;
        let result_distributed = pretty_format_batches(&batches_distributed)?;

        println!("\n=== DISTRIBUTED RESULTS ===");
        println!("{}", result_distributed);

        // Create a non-distributed context for comparison
        use datafusion::prelude::SessionContext;
        let ctx_non_distributed = SessionContext::new();

        // Enable file partitioning preservation in non-distributed context too
        ctx_non_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Set target_partitions to 4 (same as distributed context)
        ctx_non_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register the same tables in non-distributed context
        let dim_options_nd = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_non_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options_nd)
            .await?;

        let fact_options_nd = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)])
            .file_sort_order(vec![
                vec![
                    datafusion::prelude::col("f_dkey").sort(true, true),
                    datafusion::prelude::col("timestamp").sort(true, true),
                ]
            ]);
        ctx_non_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options_nd)
            .await?;

        // Execute the same query in non-distributed context
        let df_non_distributed = ctx_non_distributed.sql(query).await?;
        let physical_non_distributed = df_non_distributed.clone().create_physical_plan().await?;
        let physical_non_distributed_str = display_plan_ascii(physical_non_distributed.as_ref(), false);

        println!("\n=== NON-DISTRIBUTED PLAN ===");
        println!("{}", physical_non_distributed_str);

        let batches_non_distributed = df_non_distributed.collect().await?;
        let result_non_distributed = pretty_format_batches(&batches_non_distributed)?;

        println!("\n=== NON-DISTRIBUTED RESULTS ===");
        println!("{}", result_non_distributed);

        // Compare results: both should have the same data
        let total_rows_distributed: usize = batches_distributed.iter().map(|b| b.num_rows()).sum();
        let total_rows_non_distributed: usize = batches_non_distributed.iter().map(|b| b.num_rows()).sum();

        assert_eq!(
            total_rows_distributed, total_rows_non_distributed,
            "Row count mismatch: distributed={}, non-distributed={}",
            total_rows_distributed, total_rows_non_distributed
        );

        // Compare the actual data by converting to strings and comparing
        // Since both are sorted by (f_dkey, time_bin), they should be identical
        assert_eq!(
            result_distributed.to_string(),
            result_non_distributed.to_string(),
            "Results differ between distributed and non-distributed execution"
        );

        println!("\n✓ Verified: Distributed and non-distributed results are identical ({} rows)", total_rows_distributed);

        // Expected plan with join + date_bin aggregation (OPTIMAL WITH HASH SUPERSET):
        // Similar to the timestamp aggregation test, this achieves a SINGLE-STAGE execution plan!
        // 
        // Stage 1 ONLY (Distributed):
        //   - Collocated join using Hive partitioning (NO shuffle!)
        //   - Single-pass aggregation with date_bin (mode=SinglePartitioned) - NO repartitioning!
        //   - Sort for final ORDER BY
        // 
        // KEY OPTIMIZATION: Hash Superset Satisfies Partitioning
        // - The join is partitioned by d_dkey (via Hive partitioning)
        // - The aggregation needs (d_dkey, date_bin(timestamp))
        // - Since d_dkey is a SUPERSET of the join partitioning, NO repartition is needed!
        // - DataFusion recognizes that data partitioned by d_dkey is sufficient for aggregating by (d_dkey, date_bin(timestamp))
        // 
        // NOTE: This optimization works even with repartition_subset_satisfactions at its default value
        let expected_plan = r#"┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [f_dkey@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST]
│   [Stage 1] => NetworkCoalesceExec: output_partitions=4, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 1 ── Tasks: t0:[p0..p1] t1:[p2..p3] 
  │ SortExec: expr=[f_dkey@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST], preserve_partitioning=[true]
  │   ProjectionExec: expr=[f_dkey@0 as f_dkey, date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp)@1 as time_bin, max(j.env)@2 as max_env, max(j.value)@3 as max_value]
  │     AggregateExec: mode=SinglePartitioned, gby=[f_dkey@0 as f_dkey, date_bin(IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }, timestamp@2) as date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp)], aggr=[max(j.env), max(j.value)], ordering_mode=Sorted
  │       ProjectionExec: expr=[f_dkey@3 as f_dkey, env@0 as env, timestamp@1 as timestamp, value@2 as value]
  │         HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@1, f_dkey@2)], projection=[env@0, timestamp@2, value@3, f_dkey@4]
  │           PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │             DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, d_dkey], file_type=csv, has_header=true
  │           PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │             DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
  └──────────────────────────────────────────────────
"#;

        // Normalize paths for comparison
        let normalize_paths = |s: &str| -> String {
            s.lines()
                .map(|line| {
                    if line.contains("testdata/join_test_hive") {
                        let re = regex::Regex::new(r"[^,\s\[]*(/testdata/join_test_hive/[^,\s\]]+)").unwrap();
                        re.replace_all(line, "$1").to_string()
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        
        let normalized_expected = normalize_paths(&expected_plan);
        let normalized_actual = normalize_paths(&physical_distributed_str);
        
        assert_eq!(
            normalized_actual.trim(),
            normalized_expected.trim(),
            "\n\nActual plan does not match expected plan!\n\nExpected:\n{}\n\nActual:\n{}\n",
            normalized_expected,
            normalized_actual
        );

        // Verify we got results
        assert!(
            total_rows_distributed > 0,
            "Should have some aggregated rows, got {}",
            total_rows_distributed
        );

        // Verify that all f_dkey values are present (A, B, C, D)
        let mut f_dkey_values = std::collections::HashSet::new();
        for batch in &batches_distributed {
            let f_dkey_array = batch
                .column_by_name("f_dkey")
                .expect("f_dkey column should exist");
            
            let f_dkey_strings = datafusion::arrow::array::cast::as_string_array(f_dkey_array);
            for i in 0..f_dkey_strings.len() {
                f_dkey_values.insert(f_dkey_strings.value(i).to_string());
            }
        }

        // Verify we have all partition keys
        assert!(
            f_dkey_values.len() >= 4,
            "Should have all 4 partitions (A, B, C, D), got {:?}",
            f_dkey_values
        );

        println!("\n✓ Verified: All partitions (A, B, C, D) present in date_bin aggregated results");
        println!("✓ Total aggregated rows (30-second bins): {}", total_rows_distributed);
        println!("✓ Unique f_dkey values: {:?}", f_dkey_values);

        Ok(())
    }
}

