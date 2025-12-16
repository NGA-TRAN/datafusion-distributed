//! # Distributed Join Test with Custom Task Estimator (Mismatched Partitioning)
//!
//! This test demonstrates a scenario where custom task estimation is used but network shuffles
//! are still required due to mismatched partitioning between tables.
//!
//! ## Scenario: Custom Task Estimator with Mismatched Partitioning (REQUIRES SHUFFLES)
//! - Uses `CustomJoinTaskEstimator`
//! - Data: `testdata/join_test_hive_custom/` with mismatched partitioning
//!   - Dim: 2 files (A_B, C_D) - join keys span multiple files
//!   - Fact: 4 files (A, B, C, D) - one file per join key
//! - Result: Multi-stage join WITH network shuffles (Stage 1: dim repartition, Stage 2: fact repartition, Stage 3: join)
//! - Why: DataFusion cannot infer that data is co-located because partition keys don't match
//!
//! ## Key Takeaway
//! Custom task estimators control task distribution but CANNOT avoid shuffles when DataFusion
//! cannot infer data collocation from partition keys.

#[cfg(all(feature = "integration", test))]
mod tests {
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::catalog::memory::DataSourceExec;
    use datafusion::config::ConfigOptions;
    use datafusion::datasource::physical_plan::FileScanConfig;
    use datafusion::error::DataFusionError;
    use datafusion::execution::{SessionState, SessionStateBuilder};
    use datafusion::physical_plan::{displayable, execute_stream, ExecutionPlan};
    use datafusion::prelude::{CsvReadOptions, SessionContext};
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::{
        DistributedExt, DistributedSessionBuilderContext, PartitionIsolatorExec, TaskEstimation,
        TaskEstimator, display_plan_ascii,
    };
    use futures::TryStreamExt;
    use std::error::Error;
    use std::sync::Arc;

    /// Custom TaskEstimator that assigns different partitions per task based on table name.
    ///
    /// NOTE: This estimator controls task distribution but CANNOT avoid network shuffles
    /// when partition keys don't match between tables.
    #[derive(Debug)]
    struct CustomJoinTaskEstimator {
        dim_partitions_per_task: usize,
        fact_partitions_per_task: usize,
        num_tasks: usize,
    }

    impl CustomJoinTaskEstimator {
        fn new(dim_partitions_per_task: usize, fact_partitions_per_task: usize, num_tasks: usize) -> Self {
            Self {
                dim_partitions_per_task,
                fact_partitions_per_task,
                num_tasks,
            }
        }
    }

    impl TaskEstimator for CustomJoinTaskEstimator {
        fn estimate_tasks(
            &self,
            plan: &Arc<dyn ExecutionPlan>,
            _cfg: &ConfigOptions,
        ) -> Option<TaskEstimation> {
            // Try to downcast to DataSourceExec
            let dse: &DataSourceExec = plan.as_any().downcast_ref()?;
            let file_scan: &FileScanConfig = dse.data_source().as_any().downcast_ref()?;

            // Determine table name from the file paths
            let table_name = if let Some(first_group) = file_scan.file_groups.first() {
                if let Some(first_file) = first_group.iter().next() {
                    let path = first_file.object_meta.location.as_ref();
                    if path.contains("/dim/") {
                        "dim"
                    } else if path.contains("/fact/") {
                        "fact"
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            };

            // Calculate partitions per task based on table type
            let _partitions_per_task = match table_name {
                "dim" => self.dim_partitions_per_task,
                "fact" => self.fact_partitions_per_task,
                _ => return None,
            };
            
            let task_count = self.num_tasks;

            // Scale up partitions to match task distribution
            // IMPORTANT: For partitioned hash joins, both sides must have the same number of partitions
            // We use the maximum (fact's 4 partitions) to ensure compatibility
            let scaled_partitions = task_count * std::cmp::max(self.dim_partitions_per_task, self.fact_partitions_per_task);
            let mut plan = Arc::clone(plan);
            
            // Repartition if needed (this will add RepartitionExec if current partitions < scaled_partitions)
            if let Ok(Some(repartitioned)) = plan.repartitioned(scaled_partitions, _cfg) {
                plan = repartitioned;
            }

            // Wrap with PartitionIsolatorExec to distribute partitions across tasks
            plan = Arc::new(PartitionIsolatorExec::new(plan));

            Some(TaskEstimation {
                task_count,
                new_plan: Some(plan),
            })
        }
    }

    async fn build_state(
        ctx: DistributedSessionBuilderContext,
    ) -> Result<SessionState, DataFusionError> {
        Ok(SessionStateBuilder::new()
            .with_runtime_env(ctx.runtime_env)
            .with_default_features()
            // dim: 1 file per task, fact: 2 files per task, total: 2 tasks
            .with_distributed_task_estimator(CustomJoinTaskEstimator::new(1, 2, 2))
            .build())
    }

    #[tokio::test]
    async fn test_join_with_custom_task_estimator() -> Result<(), Box<dyn Error>> {
        // Start distributed context with 2 workers
        // Each worker will process: 1 dim file + 2 fact files and join locally
        let (ctx_distributed, _guard) = start_localhost_context(2, build_state).await;

        // NOTE: We CANNOT enable preserve_file_partitions here because:
        // - Dim has 2 file groups but fact has 4
        // - preserve_file_partitions prevents repartitioning, but we NEED to repartition
        // - The CustomTaskEstimator's repartitioned() call requires preserve_file_partitions=0
        // - This scenario demonstrates that network shuffles are sometimes necessary
        
        // Set target_partitions to match fact table (4 partitions)
        ctx_distributed.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;

        // Register dimension table
        // CUSTOM SCENARIO: Dim has 2 files (A_B.csv, C_D.csv) with join keys in the data
        ctx_distributed
            .register_csv("dim", "testdata/join_test_hive_custom/dim", CsvReadOptions::default())
            .await?;

        // Register fact table
        // Fact has 4 files (A.csv, B.csv, C.csv, D.csv) with join keys in the data
        ctx_distributed
            .register_csv("fact", "testdata/join_test_hive_custom/fact", CsvReadOptions::default())
            .await?;

        // Create a join query
        let query = r#"
            SELECT 
                d.d_dkey,
                d.env,
                d.service,
                d.host,
                f.timestamp,
                f.value
            FROM dim d
            INNER JOIN fact f ON d.d_dkey = f.f_dkey
            ORDER BY d.d_dkey, f.timestamp
        "#;

        // Create non-distributed context for comparison
        let ctx = SessionContext::default();
        *ctx.state_ref().write().config_mut() = ctx_distributed.copied_config();
        
        ctx.state_ref().write().config_mut().options_mut()
            .execution.target_partitions = 4;
        
        ctx.register_csv("dim", "testdata/join_test_hive_custom/dim", CsvReadOptions::default())
            .await?;
        
        ctx.register_csv("fact", "testdata/join_test_hive_custom/fact", CsvReadOptions::default())
            .await?;

        // Get non-distributed plan
        let df = ctx.sql(query).await?;
        let physical = df.create_physical_plan().await?;
        let _physical_str = displayable(physical.as_ref()).indent(true).to_string();

        // Get distributed plan
        let df_distributed = ctx_distributed.sql(query).await?;
        let physical_distributed = df_distributed.create_physical_plan().await?;
        let physical_distributed_str = display_plan_ascii(physical_distributed.as_ref(), false);

        // Print plans for inspection
        println!("\nDistributed plan:\n{}", physical_distributed_str);

        // Execute non-distributed query
        let batches = execute_stream(physical, ctx.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;
        let result = pretty_format_batches(&batches)?;

        // Execute distributed query
        let batches_distributed = execute_stream(physical_distributed, ctx_distributed.task_ctx())?
            .try_collect::<Vec<_>>()
            .await?;
        let result_distributed = pretty_format_batches(&batches_distributed)?;

        // Print results
        println!("\nDistributed result:\n{}", result_distributed);

        // Verify results match
        assert_eq!(
            result.to_string(),
            result_distributed.to_string(),
            "Distributed and non-distributed results should match"
        );

        // Verify we got the expected number of rows (all joins should succeed)
        let total_rows: usize = batches_distributed.iter().map(|b| b.num_rows()).sum();
        assert!(
            total_rows > 0,
            "Should have some joined rows, got {}",
            total_rows
        );

        // Expected plan for CUSTOM scenario with mismatched partitioning (updated for hash superset):
        // - Dim has 2 files (each covering multiple keys: A_B and C_D)
        // - Fact has 4 files (one per key: A, B, C, D)
        // 
        // CustomTaskEstimator distributes:
        // - Task 0: dim[A_B] (1 file) + fact[A,B] (2 files)
        // - Task 1: dim[C_D] (1 file) + fact[C,D] (2 files)
        //
        // IMPORTANT: This scenario requires network shuffles (Stage 1 and Stage 2) because:
        // 1. Dim partition keys (A_B, C_D) don't match fact partition keys (A, B, C, D)
        // 2. DataFusion cannot infer that the data is co-located by join key
        // 3. To avoid shuffles, you would need to either:
        //    - Use Hive-style partitioning with matching partition keys
        //    - Implement a custom physical optimizer rule that understands the collocation
        //    - Use a broadcast join (CollectLeft) if dim is small enough
        let expected_plan = r#"┌───── DistributedExec ── Tasks: t0:[p0] 
│ SortPreservingMergeExec: [d_dkey@0 ASC NULLS LAST, timestamp@4 ASC NULLS LAST]
│   [Stage 3] => NetworkCoalesceExec: output_partitions=8, input_tasks=2
└──────────────────────────────────────────────────
  ┌───── Stage 3 ── Tasks: t0:[p0..p3] t1:[p0..p3] 
  │ SortExec: expr=[d_dkey@0 ASC NULLS LAST, timestamp@4 ASC NULLS LAST], preserve_partitioning=[true]
  │   HashJoinExec: mode=Partitioned, join_type=Inner, on=[(d_dkey@0, f_dkey@0)], projection=[d_dkey@0, env@1, service@2, host@3, timestamp@5, value@6]
  │     [Stage 1] => NetworkShuffleExec: output_partitions=4, input_tasks=2
  │     [Stage 2] => NetworkShuffleExec: output_partitions=4, input_tasks=2
  └──────────────────────────────────────────────────
    ┌───── Stage 1 ── Tasks: t0:[p0..p7] t1:[p0..p7] 
    │ CoalesceBatchesExec: target_batch_size=8192
    │   RepartitionExec: partitioning=Hash([d_dkey@0], 8), input_partitions=1
    │     PartitionIsolatorExec: t0:[p0,__] t1:[__,p0] 
    │       DataSourceExec: file_groups={2 groups: [[/testdata/join_test_hive_custom/dim/A_B.csv], [/testdata/join_test_hive_custom/dim/C_D.csv]]}, projection=[d_dkey, env, service, host], file_type=csv, has_header=true
    └──────────────────────────────────────────────────
    ┌───── Stage 2 ── Tasks: t0:[p0..p7] t1:[p0..p7] 
    │ CoalesceBatchesExec: target_batch_size=8192
    │   RepartitionExec: partitioning=Hash([f_dkey@0], 8), input_partitions=2
    │     PartitionIsolatorExec: t0:[p0,p1,__,__] t1:[__,__,p0,p1] 
    │       DataSourceExec: file_groups={4 groups: [[/testdata/join_test_hive_custom/fact/A.csv], [/testdata/join_test_hive_custom/fact/B.csv], [/testdata/join_test_hive_custom/fact/C.csv], [/testdata/join_test_hive_custom/fact/D.csv]]}, projection=[f_dkey, timestamp, value], file_type=csv, has_header=true
    └──────────────────────────────────────────────────
"#;
        
        // Assert that actual plan matches expected plan (ignoring absolute vs relative paths)
        // Normalize paths for comparison
        let normalize_paths = |s: &str| -> String {
            s.lines()
                .map(|line| {
                    // Remove absolute path prefix, keeping only relative path from testdata/
                    if line.contains("testdata/join_test_hive") {
                        let re = regex::Regex::new(r"[^,\s\[]*(/testdata/join_test_hive[^,\s\]]+)").unwrap();
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

        Ok(())
    }
}

