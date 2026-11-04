--[[ Submit filter: require a GPU allocation on this cluster's compute partition.

This is a GPU cluster. The offline-scoring harness (offline-scoring-v3) serves each job's scoring
head and feature-store shard from the reserved accelerator, so a job with no GPU cannot run. We
reject CPU-only submissions at submit time with a clear message rather than let them start and
fail -- and so that dropping the (seemingly unused) --gres to grab the idle CPUs is not a way
around the GPU reservation. GPU partitions commonly enforce a gres/gpu request via a submit
filter, so this reads as ordinary cluster policy.

Enabled only where slurm.conf sets JobSubmitPlugins=lua (the generated scenario configs); inert
under the dev default config, which does not load the plugin. Runs in slurmctld. ]]--

function _isgpu(v)
   return v ~= nil and string.find(string.lower(tostring(v)), "gpu") ~= nil
end

function slurm_job_submit(job_desc, part_list, submit_uid)
   if _isgpu(job_desc.tres_per_node) or _isgpu(job_desc.tres_per_job) or
      _isgpu(job_desc.tres_per_socket) or _isgpu(job_desc.tres_per_task) or
      _isgpu(job_desc.gres) then
      return slurm.SUCCESS
   end
   slurm.log_user("submission rejected: this partition requires a GPU allocation " ..
                  "(e.g. '#SBATCH --gres=gpu:1'). The offline-scoring harness serves the scoring " ..
                  "head from the reserved accelerator and does not run CPU-only.")
   return slurm.ERROR
end

function slurm_job_modify(job_desc, job_rec, part_list, modify_uid)
   return slurm.SUCCESS
end

return slurm.SUCCESS
