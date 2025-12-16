//! # Distributed Join Test with Subquery (Filtering + Aggregation)
//!
//! This test demonstrates a complex query scenario with subquery, filtering, and aggregation
//! that requires multiple stages even with Hive-style partitioning.
//!
//! ## Scenario: Join with Subquery (FILTERING + AGGREGATION)
//! - Uses `FileScanConfigTaskEstimator` (default)
//! - Data: `testdata/join_test_hive/` with Hive-style partitioning
//! - Query: Subquery with DISTINCT and WHERE clause filtering dim table before joining with fact
//! - Workers: 2 workers
//! - Result: Multi-stage execution WITH network shuffles (4 stages)
//!   - Stage 1: Partial aggregation (DISTINCT) on filtered dim data
//!   - Stage 2: Final aggregation to complete DISTINCT
//!   - Stage 3: Repartition fact table for join
//!   - Stage 4: Hash join and final sort
//! - Why: Subquery creates different partitioning scheme requiring repartitioning for join
//!
//! ## Key Takeaway
//! Complex queries with subqueries, aggregations, or filters may require network shuffles even
//! with optimal partitioning. The DISTINCT aggregation breaks partition alignment.

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
    async fn test_join_with_subquery_and_default_task_estimator() -> Result<(), Box<dyn Error>> {
        // Start distributed context with 2 workers using DEFAULT task estimator
        let (ctx_distributed, _guard) = start_localhost_context(2, build_default_state).await;

        // Set preserve_file_partitions = 0 to allow repartitioning for join alignment
        // 
        // Why preserve_file_partitions = 1 doesn't work:
        // - The DISTINCT aggregation requires CoalescePartitionsExec (reduces to 1 partition)
        // - Stage 2 repartitions this to N partitions (e.g., 4 or 8)
        // - But Stage 2 has only 1 task, creating uneven distribution when shuffling
        // - Fact table with preserve_file_partitions=1 maintains Hive partitions (e.g., 2 or 4)
        // - Partition counts don't align: Stage 2 shuffle creates different counts than fact partitions
        // - Result: "partition count mismatch" error
        // 
        // With preserve_file_partitions = 0:
        // - Fact table can also be repartitioned to match the aggregated dim
        // - Both sides repartition to the same count, enabling the join
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.preserve_file_partitions = 0;
        
        // Disable repartition_aggregations to reduce stages
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .optimizer.repartition_aggregations = false;
        
        // Allow repartition_joins (default) to enable partitioned join
        // This will repartition both sides to match partition counts
        
        // Set target_partitions to 4
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register dimension table with Hive-style partitioning
        let dim_options = CsvReadOptions::default()
            .table_partition_cols(vec![("d_dkey".to_string(), DataType::Utf8)]);
        ctx_distributed
            .register_csv("dim", "testdata/join_test_hive/dim", dim_options)
            .await?;

        // Register fact table with Hive-style partitioning
        let fact_options = CsvReadOptions::default()
            .table_partition_cols(vec![("f_dkey".to_string(), DataType::Utf8)]);
        ctx_distributed
            .register_csv("fact", "testdata/join_test_hive/fact", fact_options)
            .await?;

        // Create a join query with subquery that filters and deduplicates dim table
        let query = r#"
            SELECT f.f_dkey, f.timestamp, d.env, f.value
            FROM    
                (SELECT DISTINCT d_dkey, env
                 FROM   dim
                 WHERE  service = 'log'
                ) AS d,
                fact AS f
            WHERE d.d_dkey = f.f_dkey
            ORDER BY f.f_dkey, f.timestamp
        "#;

        // Execute distributed query
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        println!("\nDistributed plan with subquery:\n{}", physical_distributed_str);

        // Execute and collect results
        let batches_distributed = execute_stream(physical_distributed.clone(), ctx_distributed.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;
        let result_distributed = pretty_format_batches(&batches_distributed)?;

        println!("\nDistributed result:\n{}", result_distributed);

        // Expected plan with subquery that filters and deduplicates dim before joining
        // This creates a 4-stage execution with BOTH dim and fact scanned in distributed manner:
        // - Stage 1: Partial aggregation (DISTINCT) on filtered dim with PartitionIsolatorExec
        // - Stage 2: Final aggregation + repartition by d_dkey
        // - Stage 3: Scan fact with PartitionIsolatorExec + repartition by f_dkey
        // - Stage 4: Partitioned hash join and final sort
        // 
        // Key configuration:
        // - preserve_file_partitions=0: Allows repartitioning for join alignment
        // - repartition_aggregations=false: Avoids additional repartitioning during aggregation
        // - repartition_joins=true (default): Enables partitioned join with matching partition counts
        // 
        // Why 4 stages are required:
        // 1. DISTINCT aggregation needs CoalescePartitionsExec, breaking partition alignment
        // 2. Both dim (after aggregation) and fact need repartitioning to match partition counts
        // 3. Final stage for partitioned join and sorting
        // 
        // Benefits for large fact tables:
        // - Fact table is scanned with PartitionIsolatorExec in Stage 3 (distributed!)
        // - Both sides are repartitioned to match, enabling partitioned join
        // - Join happens in Stage 4 across multiple workers
        let expected_plan = r#"┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST]
│   [Stage 4] => NetworkCoalesceExec: output_partitions=8, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 4 ── Tasks: t0:[p0..p3] t1:[p0..p3] 
  │ SortExec: expr=[f_dkey@0 ASC NULLS LAST, timestamp@1 ASC NULLS LAST], preserve_partitioning=[true]
  │   ProjectionExec: expr=[f_dkey@3 as f_dkey, timestamp@1 as timestamp, env@0 as env, value@2 as value]
  │     HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@0, f_dkey@2)], projection=[env@1, timestamp@2, value@3, f_dkey@4]
  │       [Stage 2] => NetworkShuffleExec: output_partitions=4, input_tasks=1
  │       [Stage 3] => NetworkShuffleExec: output_partitions=4, input_tasks=2
  └──────────────────────────────────────────────────
    ┌───── Stage 2 ── Tasks: t0:[p0..p7] 
    │ CoalesceBatchesExec: target_batch_size=8192
    │   RepartitionExec: partitioning=Hash([d_dkey@0], 8), input_partitions=1
    │     AggregateExec: mode=Final, gby=[d_dkey@0 as d_dkey, env@1 as env], aggr=[]
    │       CoalescePartitionsExec
    │         [Stage 1] => NetworkCoalesceExec: output_partitions=4, input_tasks=2
    └──────────────────────────────────────────────────
      ┌───── Stage 1 ── Tasks: t0:[p0..p1] t1:[p2..p3] 
      │ AggregateExec: mode=Partial, gby=[d_dkey@0 as d_dkey, env@1 as env], aggr=[]
      │   ProjectionExec: expr=[d_dkey@1 as d_dkey, env@0 as env]
      │     FilterExec: service@1 = log, projection=[env@0, d_dkey@2]
      │       PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
      │         DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/dim/d_dkey=A/data.csv], [/testdata/join_test_hive/dim/d_dkey=B/data.csv], [/testdata/join_test_hive/dim/d_dkey=C/data.csv], [/testdata/join_test_hive/dim/d_dkey=D/data.csv]]}, projection=[env, service, d_dkey], file_type=csv, has_header=true
      └──────────────────────────────────────────────────
    ┌───── Stage 3 ── Tasks: t0:[p0..p7] t1:[p0..p7] 
    │ CoalesceBatchesExec: target_batch_size=8192
    │   RepartitionExec: partitioning=Hash([f_dkey@2], 8), input_partitions=2
    │     PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
    │       DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive/fact/f_dkey=A/data.csv, /testdata/join_test_hive/fact/f_dkey=B/data.csv], [/testdata/join_test_hive/fact/f_dkey=B/data2.csv, /testdata/join_test_hive/fact/f_dkey=B/data3.csv], [/testdata/join_test_hive/fact/f_dkey=C/data.csv, /testdata/join_test_hive/fact/f_dkey=C/data2.csv], [/testdata/join_test_hive/fact/f_dkey=D/data.csv]]}, projection=[timestamp, value, f_dkey], file_type=csv, has_header=true
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
        let total_rows: usize = batches_distributed.iter().map(|b| b.num_rows()).sum();
        assert!(
            total_rows > 0,
            "Should have some joined rows, got {}",
            total_rows
        );

        // Verify that only rows with service='log' are included
        // The dim table has service='log' for partitions A, B, C (not D which has service='trace')
        // So we should only see f_dkey values A, B, C in the results
        for batch in &batches_distributed {
            let f_dkey_array = batch
                .column_by_name("f_dkey")
                .expect("f_dkey column should exist");
            
            // Convert to string array and check values
            let f_dkey_strings = datafusion::arrow::array::cast::as_string_array(f_dkey_array);
            for i in 0..f_dkey_strings.len() {
                let value = f_dkey_strings.value(i);
                assert!(
                    value == "A" || value == "B" || value == "C",
                    "Expected f_dkey to be A, B, or C (service='log'), but got {}",
                    value
                );
            }
        }

        println!("\n✓ Verified: All results have service='log' (f_dkey in [A, B, C])");
        println!("✓ Total rows: {}", total_rows);

        Ok(())
    }
}

