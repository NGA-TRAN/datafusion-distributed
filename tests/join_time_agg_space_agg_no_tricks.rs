//! # Test: Join with Time Aggregation then Space Aggregation (No Tricks)
//!
//! This test is similar to join_time_agg_space_agg.rs but with a simpler query that doesn't use
//! MAX(env) trick. Instead, it includes `env` directly in the GROUP BY of the inner aggregation.
//!
//! ## Query Structure:
//! 1. Inner subquery: Join dim and fact, then aggregate by (f_dkey, time_bin, env)
//! 2. Outer query: Aggregate by (env, time_bin) across partitions
//!
//! ## Key Difference from join_time_agg_space_agg.rs:
//! - Original: GROUP BY f_dkey, time_bin with MAX(env)
//! - This test: GROUP BY f_dkey, time_bin, env (no MAX trick)
//!
//! This makes the query more straightforward but may have different optimization characteristics.

#[cfg(all(feature = "integration", test))]
mod tests {
    use datafusion::arrow::array::Array;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::execution::context::SessionState;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::physical_plan::execute_stream;
    use datafusion::prelude::CsvReadOptions;
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::{DistributedSessionBuilderContext, display_plan_ascii};
    use futures::TryStreamExt;
    use std::error::Error;

    async fn build_default_state(
        ctx: DistributedSessionBuilderContext,
    ) -> Result<SessionState, datafusion::error::DataFusionError> {
        Ok(SessionStateBuilder::new()
            .with_runtime_env(ctx.runtime_env)
            .with_default_features()
            .build())
    }

    #[tokio::test]
    async fn test_join_with_time_agg_then_space_agg_no_tricks() -> Result<(), Box<dyn Error>> {
        // Start distributed context with 2 workers using DEFAULT task estimator
        let (ctx_distributed, _guard) = start_localhost_context(2, build_default_state).await;

        // Enable file partitioning preservation
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 1;

        // Enable hash subset satisfaction (from PR #19304)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.repartition_subset_satisfactions = true;

        // Set target_partitions to 4 to create 4 file groups (one per Hive partition: A, B, C, D)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Enable hash join inlist pushdown for dynamic filtering
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.hash_join_inlist_pushdown_max_size = 100;
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.hash_join_inlist_pushdown_max_distinct_values = 100;

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

        // Create a query with nested aggregations (NO TRICKS version):
        // 1. Inner: Join + time aggregation (by f_dkey, time_bin, env) - env is in GROUP BY
        // 2. Outer: Space aggregation (by env, time_bin) - aggregating across partitions
        let query = r#"
            SELECT env, time_bin, AVG(max_bin_value) AS avg_max_value
            FROM
            (
                SELECT  f_dkey, 
                        date_bin(INTERVAL '30 seconds', timestamp) AS time_bin,
                        env,
                        MAX(value) AS max_bin_value
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
                    WHERE service = 'log'
                    ) AS j
                GROUP BY f_dkey, time_bin, env
            ) AS a
            GROUP BY env, time_bin
            ORDER BY env, time_bin
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

        // Enable hash subset satisfaction for non-distributed context
        ctx_non_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.repartition_subset_satisfactions = true;

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
        assert_eq!(
            result_distributed.to_string(),
            result_non_distributed.to_string(),
            "Results differ between distributed and non-distributed execution"
        );

        println!("\n✓ Verified: Distributed and non-distributed results are identical ({} rows)", total_rows_distributed);

        // Expected distributed plan
        let expected_plan = r#"┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [env@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST]
│   SortExec: expr=[env@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST], preserve_partitioning=[true]
│     ProjectionExec: expr=[env@0 as env, time_bin@1 as time_bin, avg(a.max_bin_value)@2 as avg_max_value]
│       AggregateExec: mode=FinalPartitioned, gby=[env@0 as env, time_bin@1 as time_bin], aggr=[avg(a.max_bin_value)]
│         [Stage 1] => NetworkShuffleExec: output_partitions=4, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 1 ── Tasks: t0:[p0..p3] t1:[p0..p3] 
  │ CoalesceBatchesExec: target_batch_size=8192
  │   RepartitionExec: partitioning=Hash([env@0, time_bin@1], 4), input_partitions=2
  │     AggregateExec: mode=Partial, gby=[env@1 as env, time_bin@0 as time_bin], aggr=[avg(a.max_bin_value)]
  │       ProjectionExec: expr=[date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp)@1 as time_bin, env@2 as env, max(j.value)@3 as max_bin_value]
  │         AggregateExec: mode=SinglePartitioned, gby=[f_dkey@0 as f_dkey, date_bin(IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }, timestamp@2) as date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp), env@1 as env], aggr=[max(j.value)], ordering_mode=PartiallySorted([0, 1])
  │           ProjectionExec: expr=[f_dkey@3 as f_dkey, env@0 as env, timestamp@1 as timestamp, value@2 as value]
  │             HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@1, f_dkey@2)], projection=[env@0, timestamp@2, value@3, f_dkey@4]
  │               FilterExec: service@1 = log, projection=[env@0, d_dkey@2]
  │                 PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │                   DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, service, d_dkey], file_type=csv, has_header=true
  │               PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
  │                 DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
  └──────────────────────────────────────────────────
"#;

        // Expected non-distributed plan (with preserve_file_partitions and target_partitions=4)
        let expected_non_distributed_plan = r#"SortPreservingMergeExec: [env@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST]
  SortExec: expr=[env@0 ASC NULLS LAST, time_bin@1 ASC NULLS LAST], preserve_partitioning=[true]
    ProjectionExec: expr=[env@0 as env, time_bin@1 as time_bin, avg(a.max_bin_value)@2 as avg_max_value]
      AggregateExec: mode=FinalPartitioned, gby=[env@0 as env, time_bin@1 as time_bin], aggr=[avg(a.max_bin_value)]
        CoalesceBatchesExec: target_batch_size=8192
          RepartitionExec: partitioning=Hash([env@0, time_bin@1], 4), input_partitions=4
            AggregateExec: mode=Partial, gby=[env@1 as env, time_bin@0 as time_bin], aggr=[avg(a.max_bin_value)]
              ProjectionExec: expr=[date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp)@1 as time_bin, env@2 as env, max(j.value)@3 as max_bin_value]
                AggregateExec: mode=SinglePartitioned, gby=[f_dkey@0 as f_dkey, date_bin(IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }, timestamp@2) as date_bin(IntervalMonthDayNano("IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 30000000000 }"),j.timestamp), env@1 as env], aggr=[max(j.value)], ordering_mode=PartiallySorted([0, 1])
                  ProjectionExec: expr=[f_dkey@3 as f_dkey, env@0 as env, timestamp@1 as timestamp, value@2 as value]
                    HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@1, f_dkey@2)], projection=[env@0, timestamp@2, value@3, f_dkey@4]
                      FilterExec: service@1 = log, projection=[env@0, d_dkey@2]
                        DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, service, d_dkey], file_type=csv, has_header=true
                      DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data3.csv, /testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=C/data2.csv, /testdata/join_test_hive/fact/f_dkey=C/data.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
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
            "\n\nActual distributed plan does not match expected plan!\n\nExpected:\n{}\n\nActual:\n{}\n",
            normalized_expected,
            normalized_actual
        );

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
        assert!(
            total_rows_distributed > 0,
            "Should have some aggregated rows, got {}",
            total_rows_distributed
        );

        // Verify that we have env values
        let mut env_values = std::collections::HashSet::new();
        for batch in &batches_distributed {
            let env_array = batch
                .column_by_name("env")
                .expect("env column should exist");
            
            let env_strings = datafusion::arrow::array::cast::as_string_array(env_array);
            for i in 0..env_strings.len() {
                env_values.insert(env_strings.value(i).to_string());
            }
        }

        // Verify we have env values (dev, prod)
        assert!(
            !env_values.is_empty(),
            "Should have env values, got {:?}",
            env_values
        );

        println!("\n✓ Verified: Env values present in aggregated results: {:?}", env_values);
        println!("✓ Total aggregated rows (by env and time_bin): {}", total_rows_distributed);

        Ok(())
    }
}

