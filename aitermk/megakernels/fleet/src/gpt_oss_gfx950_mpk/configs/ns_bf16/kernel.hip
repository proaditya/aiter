// libAiterMK fleet MPK provider source, generated from this build's test.cu by mirage.mpk.provider_export.
// Device addresses come from the pointer table fleet_mpk_init() receives; the task graph is compiled in.
static void *const *mpk_provider_ptrs = nullptr;
static const char *mpk_provider_task_graph_json(void);
// Include standard C++ headers first (before persistent_kernel.cuh)
// This avoids namespace conflicts with HIP's CUDA wrapper headers
#include <vector>
#include <thread>
#include <unistd.h>
#define MPK_USE_CK_FMHA 1
#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace mirage::runtime;
size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

void construct_task_graph(int num_gpus,
                          int my_gpu_id,
                          std::vector<FullTaskDesc> &all_tasks,
                          std::vector<EventDesc> &all_events,
                          std::vector<TaskId> &first_tasks,
                          std::map<std::string, void*> const &all_tensors) {
  nlohmann::json json_task_graph = nlohmann::json::parse(mpk_provider_task_graph_json());
  for (json const &task : json_task_graph["all_tasks"]) {
    FullTaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    task_desc.task_metadata.request_id = task.at("request_id").get<int>();
    task_desc.task_metadata.expert_offset = task.at("expert_offset").get<int>();
    task_desc.task_metadata.kv_idx = task.at("kv_idx").get<int>();
    task_desc.task_metadata.merge_task_offset = task.at("merge_task_offset").get<int>();
    task_desc.task_metadata.n_tile_start = task.value("n_tile_start", 0);
    task_desc.task_metadata.n_tile_count = task.value("n_tile_count", 0);
    if (task.at("trigger_event").is_number_integer()) {
      task_desc.trigger_event = task.at("trigger_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    task_desc.num_inputs = 0;
    for (json const &tensor : task["inputs"]) {
      TensorDesc input;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      input.num_dims = tensor.at("dims").size();
      input.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < input.num_dims; i++) {
        input.dim[i] = tensor["dims"][i].get<int>();
        input.stride[i] = tensor["strides"][i].get<int>();
      }
      task_desc.inputs[task_desc.num_inputs++] = input;
    }
    task_desc.num_outputs = 0;
    for (json const &tensor : task["outputs"]) {
      TensorDesc output;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      output.num_dims = tensor.at("dims").size();
      output.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < output.num_dims; i++) {
        output.dim[i] = tensor["dims"][i];
        output.stride[i] = tensor["strides"][i];
      }
      task_desc.outputs[task_desc.num_outputs++] = output;
    }
    #ifdef MPK_ENABLE_TMA
    if (task.at("task_type") > TASK_HOPPER_TASK_BEGIN && task.at("task_type") < TASK_HOPPER_TASK_END) {
      create_tma_desc_by_task(task_desc);
    }
    if (task.at("task_type") > TASK_SM100_TMA_START_TASK && task.at("task_type") < TASK_SM100_TMA_END_TASK) {
      create_tma_desc_by_task(task_desc);
    }
    #endif
    all_tasks.push_back(task_desc);
  }
  for (json const &e : json_task_graph["all_events"]) {
    EventType event_type = static_cast<EventType>(e.at("event_type").get<int>());
    int num_triggers = e.at("num_triggers").get<int>();
    int first_task_id = e.at("first_task_id").get<int>();
    int last_task_id = e.at("last_task_id").get<int>();
    all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));
  }
  for (json const &t : json_task_graph["first_tasks"]) {
    first_tasks.push_back(t.get<int>());
  }
}

static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                  std::vector<TaskId> &first_tasks,
                                  int num_gpus,
                                  int my_gpu_id) {
  assert(num_gpus == 1);
  std::map<std::string, void*> all_tensors;
  char *input_token = (char*)(mpk_provider_ptrs[0]);
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = (char*)(mpk_provider_ptrs[1]);
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = (char*)(mpk_provider_ptrs[2]);
  all_tensors["sin_position_embedding"] = sin_position_embedding;
  char *embed_out = (char*)(mpk_provider_ptrs[3]);
  all_tensors["embed_out"] = embed_out;
  char *rmsnorm_out = (char*)(mpk_provider_ptrs[4]);
  all_tensors["rmsnorm_out"] = rmsnorm_out;
  char *attn_in = (char*)(mpk_provider_ptrs[5]);
  all_tensors["attn_in"] = attn_in;
  char *ck_fmha_q_workspace = (char*)(mpk_provider_ptrs[6]);
  all_tensors["ck_fmha_q_workspace"] = ck_fmha_q_workspace;
  char *ck_fmha_lse_acc = (char*)(mpk_provider_ptrs[7]);
  all_tensors["ck_fmha_lse_acc"] = ck_fmha_lse_acc;
  char *attn_out = (char*)(mpk_provider_ptrs[8]);
  all_tensors["attn_out"] = attn_out;
  char *ck_fmha_o_acc = (char*)(mpk_provider_ptrs[9]);
  all_tensors["ck_fmha_o_acc"] = ck_fmha_o_acc;
  char *attn_proj_out = (char*)(mpk_provider_ptrs[10]);
  all_tensors["attn_proj_out"] = attn_proj_out;
  char *all_reduce_buf = (char*)(mpk_provider_ptrs[11]);
  all_tensors["all_reduce_buf"] = all_reduce_buf;
  char *attn_allreduce_out = (char*)(mpk_provider_ptrs[12]);
  all_tensors["attn_allreduce_out"] = attn_allreduce_out;
  char *rmsnorm_out_moe = (char*)(mpk_provider_ptrs[13]);
  all_tensors["rmsnorm_out_moe"] = rmsnorm_out_moe;
  char *moe_gate_out = (char*)(mpk_provider_ptrs[14]);
  all_tensors["moe_gate_out"] = moe_gate_out;
  char *moe_routing_indices = (char*)(mpk_provider_ptrs[15]);
  all_tensors["moe_routing_indices"] = moe_routing_indices;
  char *moe_mask = (char*)(mpk_provider_ptrs[16]);
  all_tensors["moe_mask"] = moe_mask;
  char *moe_topk_weight = (char*)(mpk_provider_ptrs[17]);
  all_tensors["moe_topk_weight"] = moe_topk_weight;
  char *router_topk_counter = (char*)(mpk_provider_ptrs[18]);
  all_tensors["router_topk_counter"] = router_topk_counter;
  char *oproj_topk_counters = (char*)(mpk_provider_ptrs[19]);
  all_tensors["oproj_topk_counters"] = oproj_topk_counters;
  char *qkv_attn_barrier = (char*)(mpk_provider_ptrs[20]);
  all_tensors["qkv_attn_barrier"] = qkv_attn_barrier;
  char *moe_fused_barrier = (char*)(mpk_provider_ptrs[21]);
  all_tensors["moe_fused_barrier"] = moe_fused_barrier;
  char *swiglu_out = (char*)(mpk_provider_ptrs[22]);
  all_tensors["swiglu_out"] = swiglu_out;
  char *mlp_out = (char*)(mpk_provider_ptrs[23]);
  all_tensors["mlp_out"] = mlp_out;
  char *moe_workspace_f32 = (char*)(mpk_provider_ptrs[24]);
  all_tensors["moe_workspace_f32"] = moe_workspace_f32;
  char *moe_ws_zero = (char*)(mpk_provider_ptrs[25]);
  all_tensors["moe_ws_zero"] = moe_ws_zero;
  char *mlp_weighted_sum_out = (char*)(mpk_provider_ptrs[26]);
  all_tensors["mlp_weighted_sum_out"] = mlp_weighted_sum_out;
  char *mlp_final = (char*)(mpk_provider_ptrs[27]);
  all_tensors["mlp_final"] = mlp_final;
  char *argmax_part_value = (char*)(mpk_provider_ptrs[28]);
  all_tensors["argmax_part_value"] = argmax_part_value;
  char *argmax_part_index = (char*)(mpk_provider_ptrs[29]);
  all_tensors["argmax_part_index"] = argmax_part_index;
  char *output_token = (char*)(mpk_provider_ptrs[30]);
  all_tensors["output_token"] = output_token;
  char *splitk_workspace = (char*)(mpk_provider_ptrs[31]);
  all_tensors["splitk_workspace"] = splitk_workspace;
  char *splitk_done_counter = (char*)(mpk_provider_ptrs[32]);
  all_tensors["splitk_done_counter"] = splitk_done_counter;
  char *embed_tokens = (char*)(mpk_provider_ptrs[33]);
  all_tensors["embed_tokens"] = embed_tokens;
  char *moe_workspace_f32_r1 = (char*)(mpk_provider_ptrs[34]);
  all_tensors["moe_workspace_f32_r1"] = moe_workspace_f32_r1;
  char *moe_workspace_f32_r2 = (char*)(mpk_provider_ptrs[35]);
  all_tensors["moe_workspace_f32_r2"] = moe_workspace_f32_r2;
  char *moe_workspace_f32_r3 = (char*)(mpk_provider_ptrs[36]);
  all_tensors["moe_workspace_f32_r3"] = moe_workspace_f32_r3;
  char *moe_workspace_f32_r4 = (char*)(mpk_provider_ptrs[37]);
  all_tensors["moe_workspace_f32_r4"] = moe_workspace_f32_r4;
  char *moe_workspace_f32_r5 = (char*)(mpk_provider_ptrs[38]);
  all_tensors["moe_workspace_f32_r5"] = moe_workspace_f32_r5;
  char *moe_workspace_f32_r6 = (char*)(mpk_provider_ptrs[39]);
  all_tensors["moe_workspace_f32_r6"] = moe_workspace_f32_r6;
  char *moe_workspace_f32_r7 = (char*)(mpk_provider_ptrs[40]);
  all_tensors["moe_workspace_f32_r7"] = moe_workspace_f32_r7;
  char *moe_workspace_f32_r8 = (char*)(mpk_provider_ptrs[41]);
  all_tensors["moe_workspace_f32_r8"] = moe_workspace_f32_r8;
  char *moe_workspace_f32_r9 = (char*)(mpk_provider_ptrs[42]);
  all_tensors["moe_workspace_f32_r9"] = moe_workspace_f32_r9;
  char *moe_workspace_f32_r10 = (char*)(mpk_provider_ptrs[43]);
  all_tensors["moe_workspace_f32_r10"] = moe_workspace_f32_r10;
  char *moe_workspace_f32_r11 = (char*)(mpk_provider_ptrs[44]);
  all_tensors["moe_workspace_f32_r11"] = moe_workspace_f32_r11;
  char *moe_workspace_f32_r12 = (char*)(mpk_provider_ptrs[45]);
  all_tensors["moe_workspace_f32_r12"] = moe_workspace_f32_r12;
  char *moe_workspace_f32_r13 = (char*)(mpk_provider_ptrs[46]);
  all_tensors["moe_workspace_f32_r13"] = moe_workspace_f32_r13;
  char *moe_workspace_f32_r14 = (char*)(mpk_provider_ptrs[47]);
  all_tensors["moe_workspace_f32_r14"] = moe_workspace_f32_r14;
  char *moe_workspace_f32_r15 = (char*)(mpk_provider_ptrs[48]);
  all_tensors["moe_workspace_f32_r15"] = moe_workspace_f32_r15;
  char *moe_workspace_f32_r16 = (char*)(mpk_provider_ptrs[49]);
  all_tensors["moe_workspace_f32_r16"] = moe_workspace_f32_r16;
  char *moe_workspace_f32_r17 = (char*)(mpk_provider_ptrs[50]);
  all_tensors["moe_workspace_f32_r17"] = moe_workspace_f32_r17;
  char *moe_workspace_f32_r18 = (char*)(mpk_provider_ptrs[51]);
  all_tensors["moe_workspace_f32_r18"] = moe_workspace_f32_r18;
  char *moe_workspace_f32_r19 = (char*)(mpk_provider_ptrs[52]);
  all_tensors["moe_workspace_f32_r19"] = moe_workspace_f32_r19;
  char *moe_workspace_f32_r20 = (char*)(mpk_provider_ptrs[53]);
  all_tensors["moe_workspace_f32_r20"] = moe_workspace_f32_r20;
  char *moe_workspace_f32_r21 = (char*)(mpk_provider_ptrs[54]);
  all_tensors["moe_workspace_f32_r21"] = moe_workspace_f32_r21;
  char *moe_workspace_f32_r22 = (char*)(mpk_provider_ptrs[55]);
  all_tensors["moe_workspace_f32_r22"] = moe_workspace_f32_r22;
  char *moe_workspace_f32_r23 = (char*)(mpk_provider_ptrs[56]);
  all_tensors["moe_workspace_f32_r23"] = moe_workspace_f32_r23;
  char *moe_workspace_f32_r24 = (char*)(mpk_provider_ptrs[57]);
  all_tensors["moe_workspace_f32_r24"] = moe_workspace_f32_r24;
  char *moe_workspace_f32_r25 = (char*)(mpk_provider_ptrs[58]);
  all_tensors["moe_workspace_f32_r25"] = moe_workspace_f32_r25;
  char *moe_workspace_f32_r26 = (char*)(mpk_provider_ptrs[59]);
  all_tensors["moe_workspace_f32_r26"] = moe_workspace_f32_r26;
  char *moe_workspace_f32_r27 = (char*)(mpk_provider_ptrs[60]);
  all_tensors["moe_workspace_f32_r27"] = moe_workspace_f32_r27;
  char *moe_workspace_f32_r28 = (char*)(mpk_provider_ptrs[61]);
  all_tensors["moe_workspace_f32_r28"] = moe_workspace_f32_r28;
  char *moe_workspace_f32_r29 = (char*)(mpk_provider_ptrs[62]);
  all_tensors["moe_workspace_f32_r29"] = moe_workspace_f32_r29;
  char *moe_workspace_f32_r30 = (char*)(mpk_provider_ptrs[63]);
  all_tensors["moe_workspace_f32_r30"] = moe_workspace_f32_r30;
  char *moe_workspace_f32_r31 = (char*)(mpk_provider_ptrs[64]);
  all_tensors["moe_workspace_f32_r31"] = moe_workspace_f32_r31;
  char *moe_workspace_f32_r32 = (char*)(mpk_provider_ptrs[65]);
  all_tensors["moe_workspace_f32_r32"] = moe_workspace_f32_r32;
  char *moe_workspace_f32_r33 = (char*)(mpk_provider_ptrs[66]);
  all_tensors["moe_workspace_f32_r33"] = moe_workspace_f32_r33;
  char *moe_workspace_f32_r34 = (char*)(mpk_provider_ptrs[67]);
  all_tensors["moe_workspace_f32_r34"] = moe_workspace_f32_r34;
  char *moe_workspace_f32_r35 = (char*)(mpk_provider_ptrs[68]);
  all_tensors["moe_workspace_f32_r35"] = moe_workspace_f32_r35;
  char *moe_workspace_f32_r36 = (char*)(mpk_provider_ptrs[69]);
  all_tensors["moe_workspace_f32_r36"] = moe_workspace_f32_r36;
  char *attn_out_r1 = (char*)(mpk_provider_ptrs[70]);
  all_tensors["attn_out_r1"] = attn_out_r1;
  char *attn_out_r2 = (char*)(mpk_provider_ptrs[71]);
  all_tensors["attn_out_r2"] = attn_out_r2;
  char *attn_out_r3 = (char*)(mpk_provider_ptrs[72]);
  all_tensors["attn_out_r3"] = attn_out_r3;
  char *attn_out_r4 = (char*)(mpk_provider_ptrs[73]);
  all_tensors["attn_out_r4"] = attn_out_r4;
  char *attn_out_r5 = (char*)(mpk_provider_ptrs[74]);
  all_tensors["attn_out_r5"] = attn_out_r5;
  char *attn_out_r6 = (char*)(mpk_provider_ptrs[75]);
  all_tensors["attn_out_r6"] = attn_out_r6;
  char *attn_out_r7 = (char*)(mpk_provider_ptrs[76]);
  all_tensors["attn_out_r7"] = attn_out_r7;
  char *attn_out_r8 = (char*)(mpk_provider_ptrs[77]);
  all_tensors["attn_out_r8"] = attn_out_r8;
  char *attn_out_r9 = (char*)(mpk_provider_ptrs[78]);
  all_tensors["attn_out_r9"] = attn_out_r9;
  char *attn_out_r10 = (char*)(mpk_provider_ptrs[79]);
  all_tensors["attn_out_r10"] = attn_out_r10;
  char *attn_out_r11 = (char*)(mpk_provider_ptrs[80]);
  all_tensors["attn_out_r11"] = attn_out_r11;
  char *attn_out_r12 = (char*)(mpk_provider_ptrs[81]);
  all_tensors["attn_out_r12"] = attn_out_r12;
  char *attn_out_r13 = (char*)(mpk_provider_ptrs[82]);
  all_tensors["attn_out_r13"] = attn_out_r13;
  char *attn_out_r14 = (char*)(mpk_provider_ptrs[83]);
  all_tensors["attn_out_r14"] = attn_out_r14;
  char *attn_out_r15 = (char*)(mpk_provider_ptrs[84]);
  all_tensors["attn_out_r15"] = attn_out_r15;
  char *attn_out_r16 = (char*)(mpk_provider_ptrs[85]);
  all_tensors["attn_out_r16"] = attn_out_r16;
  char *attn_out_r17 = (char*)(mpk_provider_ptrs[86]);
  all_tensors["attn_out_r17"] = attn_out_r17;
  char *attn_out_r18 = (char*)(mpk_provider_ptrs[87]);
  all_tensors["attn_out_r18"] = attn_out_r18;
  char *attn_out_r19 = (char*)(mpk_provider_ptrs[88]);
  all_tensors["attn_out_r19"] = attn_out_r19;
  char *attn_out_r20 = (char*)(mpk_provider_ptrs[89]);
  all_tensors["attn_out_r20"] = attn_out_r20;
  char *attn_out_r21 = (char*)(mpk_provider_ptrs[90]);
  all_tensors["attn_out_r21"] = attn_out_r21;
  char *attn_out_r22 = (char*)(mpk_provider_ptrs[91]);
  all_tensors["attn_out_r22"] = attn_out_r22;
  char *attn_out_r23 = (char*)(mpk_provider_ptrs[92]);
  all_tensors["attn_out_r23"] = attn_out_r23;
  char *attn_out_r24 = (char*)(mpk_provider_ptrs[93]);
  all_tensors["attn_out_r24"] = attn_out_r24;
  char *attn_out_r25 = (char*)(mpk_provider_ptrs[94]);
  all_tensors["attn_out_r25"] = attn_out_r25;
  char *attn_out_r26 = (char*)(mpk_provider_ptrs[95]);
  all_tensors["attn_out_r26"] = attn_out_r26;
  char *attn_out_r27 = (char*)(mpk_provider_ptrs[96]);
  all_tensors["attn_out_r27"] = attn_out_r27;
  char *attn_out_r28 = (char*)(mpk_provider_ptrs[97]);
  all_tensors["attn_out_r28"] = attn_out_r28;
  char *attn_out_r29 = (char*)(mpk_provider_ptrs[98]);
  all_tensors["attn_out_r29"] = attn_out_r29;
  char *attn_out_r30 = (char*)(mpk_provider_ptrs[99]);
  all_tensors["attn_out_r30"] = attn_out_r30;
  char *attn_out_r31 = (char*)(mpk_provider_ptrs[100]);
  all_tensors["attn_out_r31"] = attn_out_r31;
  char *attn_out_r32 = (char*)(mpk_provider_ptrs[101]);
  all_tensors["attn_out_r32"] = attn_out_r32;
  char *attn_out_r33 = (char*)(mpk_provider_ptrs[102]);
  all_tensors["attn_out_r33"] = attn_out_r33;
  char *attn_out_r34 = (char*)(mpk_provider_ptrs[103]);
  all_tensors["attn_out_r34"] = attn_out_r34;
  char *attn_out_r35 = (char*)(mpk_provider_ptrs[104]);
  all_tensors["attn_out_r35"] = attn_out_r35;
  char *attn_proj_out_r1 = (char*)(mpk_provider_ptrs[105]);
  all_tensors["attn_proj_out_r1"] = attn_proj_out_r1;
  char *attn_proj_out_r2 = (char*)(mpk_provider_ptrs[106]);
  all_tensors["attn_proj_out_r2"] = attn_proj_out_r2;
  char *attn_proj_out_r3 = (char*)(mpk_provider_ptrs[107]);
  all_tensors["attn_proj_out_r3"] = attn_proj_out_r3;
  char *attn_proj_out_r4 = (char*)(mpk_provider_ptrs[108]);
  all_tensors["attn_proj_out_r4"] = attn_proj_out_r4;
  char *attn_proj_out_r5 = (char*)(mpk_provider_ptrs[109]);
  all_tensors["attn_proj_out_r5"] = attn_proj_out_r5;
  char *attn_proj_out_r6 = (char*)(mpk_provider_ptrs[110]);
  all_tensors["attn_proj_out_r6"] = attn_proj_out_r6;
  char *attn_proj_out_r7 = (char*)(mpk_provider_ptrs[111]);
  all_tensors["attn_proj_out_r7"] = attn_proj_out_r7;
  char *attn_proj_out_r8 = (char*)(mpk_provider_ptrs[112]);
  all_tensors["attn_proj_out_r8"] = attn_proj_out_r8;
  char *attn_proj_out_r9 = (char*)(mpk_provider_ptrs[113]);
  all_tensors["attn_proj_out_r9"] = attn_proj_out_r9;
  char *attn_proj_out_r10 = (char*)(mpk_provider_ptrs[114]);
  all_tensors["attn_proj_out_r10"] = attn_proj_out_r10;
  char *attn_proj_out_r11 = (char*)(mpk_provider_ptrs[115]);
  all_tensors["attn_proj_out_r11"] = attn_proj_out_r11;
  char *attn_proj_out_r12 = (char*)(mpk_provider_ptrs[116]);
  all_tensors["attn_proj_out_r12"] = attn_proj_out_r12;
  char *attn_proj_out_r13 = (char*)(mpk_provider_ptrs[117]);
  all_tensors["attn_proj_out_r13"] = attn_proj_out_r13;
  char *attn_proj_out_r14 = (char*)(mpk_provider_ptrs[118]);
  all_tensors["attn_proj_out_r14"] = attn_proj_out_r14;
  char *attn_proj_out_r15 = (char*)(mpk_provider_ptrs[119]);
  all_tensors["attn_proj_out_r15"] = attn_proj_out_r15;
  char *attn_proj_out_r16 = (char*)(mpk_provider_ptrs[120]);
  all_tensors["attn_proj_out_r16"] = attn_proj_out_r16;
  char *attn_proj_out_r17 = (char*)(mpk_provider_ptrs[121]);
  all_tensors["attn_proj_out_r17"] = attn_proj_out_r17;
  char *attn_proj_out_r18 = (char*)(mpk_provider_ptrs[122]);
  all_tensors["attn_proj_out_r18"] = attn_proj_out_r18;
  char *attn_proj_out_r19 = (char*)(mpk_provider_ptrs[123]);
  all_tensors["attn_proj_out_r19"] = attn_proj_out_r19;
  char *attn_proj_out_r20 = (char*)(mpk_provider_ptrs[124]);
  all_tensors["attn_proj_out_r20"] = attn_proj_out_r20;
  char *attn_proj_out_r21 = (char*)(mpk_provider_ptrs[125]);
  all_tensors["attn_proj_out_r21"] = attn_proj_out_r21;
  char *attn_proj_out_r22 = (char*)(mpk_provider_ptrs[126]);
  all_tensors["attn_proj_out_r22"] = attn_proj_out_r22;
  char *attn_proj_out_r23 = (char*)(mpk_provider_ptrs[127]);
  all_tensors["attn_proj_out_r23"] = attn_proj_out_r23;
  char *attn_proj_out_r24 = (char*)(mpk_provider_ptrs[128]);
  all_tensors["attn_proj_out_r24"] = attn_proj_out_r24;
  char *attn_proj_out_r25 = (char*)(mpk_provider_ptrs[129]);
  all_tensors["attn_proj_out_r25"] = attn_proj_out_r25;
  char *attn_proj_out_r26 = (char*)(mpk_provider_ptrs[130]);
  all_tensors["attn_proj_out_r26"] = attn_proj_out_r26;
  char *attn_proj_out_r27 = (char*)(mpk_provider_ptrs[131]);
  all_tensors["attn_proj_out_r27"] = attn_proj_out_r27;
  char *attn_proj_out_r28 = (char*)(mpk_provider_ptrs[132]);
  all_tensors["attn_proj_out_r28"] = attn_proj_out_r28;
  char *attn_proj_out_r29 = (char*)(mpk_provider_ptrs[133]);
  all_tensors["attn_proj_out_r29"] = attn_proj_out_r29;
  char *attn_proj_out_r30 = (char*)(mpk_provider_ptrs[134]);
  all_tensors["attn_proj_out_r30"] = attn_proj_out_r30;
  char *attn_proj_out_r31 = (char*)(mpk_provider_ptrs[135]);
  all_tensors["attn_proj_out_r31"] = attn_proj_out_r31;
  char *attn_proj_out_r32 = (char*)(mpk_provider_ptrs[136]);
  all_tensors["attn_proj_out_r32"] = attn_proj_out_r32;
  char *attn_proj_out_r33 = (char*)(mpk_provider_ptrs[137]);
  all_tensors["attn_proj_out_r33"] = attn_proj_out_r33;
  char *attn_proj_out_r34 = (char*)(mpk_provider_ptrs[138]);
  all_tensors["attn_proj_out_r34"] = attn_proj_out_r34;
  char *attn_proj_out_r35 = (char*)(mpk_provider_ptrs[139]);
  all_tensors["attn_proj_out_r35"] = attn_proj_out_r35;
  char *layer_0_input_layernorm = (char*)(mpk_provider_ptrs[140]);
  all_tensors["layer_0_input_layernorm"] = layer_0_input_layernorm;
  char *layer_0_qkv_mxfp4 = (char*)(mpk_provider_ptrs[141]);
  all_tensors["layer_0_qkv_mxfp4"] = layer_0_qkv_mxfp4;
  char *layer_0_qkv_bias = (char*)(mpk_provider_ptrs[142]);
  all_tensors["layer_0_qkv_bias"] = layer_0_qkv_bias;
  char *layer_0_k_cache = (char*)(mpk_provider_ptrs[143]);
  all_tensors["layer_0_k_cache"] = layer_0_k_cache;
  char *layer_0_v_cache = (char*)(mpk_provider_ptrs[144]);
  all_tensors["layer_0_v_cache"] = layer_0_v_cache;
  char *layer_0_sinks = (char*)(mpk_provider_ptrs[145]);
  all_tensors["layer_0_sinks"] = layer_0_sinks;
  char *layer_0_o_bias = (char*)(mpk_provider_ptrs[146]);
  all_tensors["layer_0_o_bias"] = layer_0_o_bias;
  char *layer_0_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[147]);
  all_tensors["layer_0_o_proj_mxfp4"] = layer_0_o_proj_mxfp4;
  char *layer_0_post_attn_layernorm = (char*)(mpk_provider_ptrs[148]);
  all_tensors["layer_0_post_attn_layernorm"] = layer_0_post_attn_layernorm;
  char *layer_0_moe_gate = (char*)(mpk_provider_ptrs[149]);
  all_tensors["layer_0_moe_gate"] = layer_0_moe_gate;
  char *layer_0_router_bias = (char*)(mpk_provider_ptrs[150]);
  all_tensors["layer_0_router_bias"] = layer_0_router_bias;
  char *layer_0_gate_up_proj = (char*)(mpk_provider_ptrs[151]);
  all_tensors["layer_0_gate_up_proj"] = layer_0_gate_up_proj;
  char *layer_0_gate_up_bias = (char*)(mpk_provider_ptrs[152]);
  all_tensors["layer_0_gate_up_bias"] = layer_0_gate_up_bias;
  char *layer_0_down_proj = (char*)(mpk_provider_ptrs[153]);
  all_tensors["layer_0_down_proj"] = layer_0_down_proj;
  char *layer_0_down_bias = (char*)(mpk_provider_ptrs[154]);
  all_tensors["layer_0_down_bias"] = layer_0_down_bias;
  char *layer_1_input_layernorm = (char*)(mpk_provider_ptrs[155]);
  all_tensors["layer_1_input_layernorm"] = layer_1_input_layernorm;
  char *layer_1_qkv_mxfp4 = (char*)(mpk_provider_ptrs[156]);
  all_tensors["layer_1_qkv_mxfp4"] = layer_1_qkv_mxfp4;
  char *layer_1_qkv_bias = (char*)(mpk_provider_ptrs[157]);
  all_tensors["layer_1_qkv_bias"] = layer_1_qkv_bias;
  char *layer_1_k_cache = (char*)(mpk_provider_ptrs[158]);
  all_tensors["layer_1_k_cache"] = layer_1_k_cache;
  char *layer_1_v_cache = (char*)(mpk_provider_ptrs[159]);
  all_tensors["layer_1_v_cache"] = layer_1_v_cache;
  char *layer_1_sinks = (char*)(mpk_provider_ptrs[160]);
  all_tensors["layer_1_sinks"] = layer_1_sinks;
  char *layer_1_o_bias = (char*)(mpk_provider_ptrs[161]);
  all_tensors["layer_1_o_bias"] = layer_1_o_bias;
  char *layer_1_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[162]);
  all_tensors["layer_1_o_proj_mxfp4"] = layer_1_o_proj_mxfp4;
  char *layer_1_post_attn_layernorm = (char*)(mpk_provider_ptrs[163]);
  all_tensors["layer_1_post_attn_layernorm"] = layer_1_post_attn_layernorm;
  char *layer_1_moe_gate = (char*)(mpk_provider_ptrs[164]);
  all_tensors["layer_1_moe_gate"] = layer_1_moe_gate;
  char *layer_1_router_bias = (char*)(mpk_provider_ptrs[165]);
  all_tensors["layer_1_router_bias"] = layer_1_router_bias;
  char *layer_1_gate_up_proj = (char*)(mpk_provider_ptrs[166]);
  all_tensors["layer_1_gate_up_proj"] = layer_1_gate_up_proj;
  char *layer_1_gate_up_bias = (char*)(mpk_provider_ptrs[167]);
  all_tensors["layer_1_gate_up_bias"] = layer_1_gate_up_bias;
  char *layer_1_down_proj = (char*)(mpk_provider_ptrs[168]);
  all_tensors["layer_1_down_proj"] = layer_1_down_proj;
  char *layer_1_down_bias = (char*)(mpk_provider_ptrs[169]);
  all_tensors["layer_1_down_bias"] = layer_1_down_bias;
  char *layer_2_input_layernorm = (char*)(mpk_provider_ptrs[170]);
  all_tensors["layer_2_input_layernorm"] = layer_2_input_layernorm;
  char *layer_2_qkv_mxfp4 = (char*)(mpk_provider_ptrs[171]);
  all_tensors["layer_2_qkv_mxfp4"] = layer_2_qkv_mxfp4;
  char *layer_2_qkv_bias = (char*)(mpk_provider_ptrs[172]);
  all_tensors["layer_2_qkv_bias"] = layer_2_qkv_bias;
  char *layer_2_k_cache = (char*)(mpk_provider_ptrs[173]);
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = (char*)(mpk_provider_ptrs[174]);
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_sinks = (char*)(mpk_provider_ptrs[175]);
  all_tensors["layer_2_sinks"] = layer_2_sinks;
  char *layer_2_o_bias = (char*)(mpk_provider_ptrs[176]);
  all_tensors["layer_2_o_bias"] = layer_2_o_bias;
  char *layer_2_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[177]);
  all_tensors["layer_2_o_proj_mxfp4"] = layer_2_o_proj_mxfp4;
  char *layer_2_post_attn_layernorm = (char*)(mpk_provider_ptrs[178]);
  all_tensors["layer_2_post_attn_layernorm"] = layer_2_post_attn_layernorm;
  char *layer_2_moe_gate = (char*)(mpk_provider_ptrs[179]);
  all_tensors["layer_2_moe_gate"] = layer_2_moe_gate;
  char *layer_2_router_bias = (char*)(mpk_provider_ptrs[180]);
  all_tensors["layer_2_router_bias"] = layer_2_router_bias;
  char *layer_2_gate_up_proj = (char*)(mpk_provider_ptrs[181]);
  all_tensors["layer_2_gate_up_proj"] = layer_2_gate_up_proj;
  char *layer_2_gate_up_bias = (char*)(mpk_provider_ptrs[182]);
  all_tensors["layer_2_gate_up_bias"] = layer_2_gate_up_bias;
  char *layer_2_down_proj = (char*)(mpk_provider_ptrs[183]);
  all_tensors["layer_2_down_proj"] = layer_2_down_proj;
  char *layer_2_down_bias = (char*)(mpk_provider_ptrs[184]);
  all_tensors["layer_2_down_bias"] = layer_2_down_bias;
  char *layer_3_input_layernorm = (char*)(mpk_provider_ptrs[185]);
  all_tensors["layer_3_input_layernorm"] = layer_3_input_layernorm;
  char *layer_3_qkv_mxfp4 = (char*)(mpk_provider_ptrs[186]);
  all_tensors["layer_3_qkv_mxfp4"] = layer_3_qkv_mxfp4;
  char *layer_3_qkv_bias = (char*)(mpk_provider_ptrs[187]);
  all_tensors["layer_3_qkv_bias"] = layer_3_qkv_bias;
  char *layer_3_k_cache = (char*)(mpk_provider_ptrs[188]);
  all_tensors["layer_3_k_cache"] = layer_3_k_cache;
  char *layer_3_v_cache = (char*)(mpk_provider_ptrs[189]);
  all_tensors["layer_3_v_cache"] = layer_3_v_cache;
  char *layer_3_sinks = (char*)(mpk_provider_ptrs[190]);
  all_tensors["layer_3_sinks"] = layer_3_sinks;
  char *layer_3_o_bias = (char*)(mpk_provider_ptrs[191]);
  all_tensors["layer_3_o_bias"] = layer_3_o_bias;
  char *layer_3_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[192]);
  all_tensors["layer_3_o_proj_mxfp4"] = layer_3_o_proj_mxfp4;
  char *layer_3_post_attn_layernorm = (char*)(mpk_provider_ptrs[193]);
  all_tensors["layer_3_post_attn_layernorm"] = layer_3_post_attn_layernorm;
  char *layer_3_moe_gate = (char*)(mpk_provider_ptrs[194]);
  all_tensors["layer_3_moe_gate"] = layer_3_moe_gate;
  char *layer_3_router_bias = (char*)(mpk_provider_ptrs[195]);
  all_tensors["layer_3_router_bias"] = layer_3_router_bias;
  char *layer_3_gate_up_proj = (char*)(mpk_provider_ptrs[196]);
  all_tensors["layer_3_gate_up_proj"] = layer_3_gate_up_proj;
  char *layer_3_gate_up_bias = (char*)(mpk_provider_ptrs[197]);
  all_tensors["layer_3_gate_up_bias"] = layer_3_gate_up_bias;
  char *layer_3_down_proj = (char*)(mpk_provider_ptrs[198]);
  all_tensors["layer_3_down_proj"] = layer_3_down_proj;
  char *layer_3_down_bias = (char*)(mpk_provider_ptrs[199]);
  all_tensors["layer_3_down_bias"] = layer_3_down_bias;
  char *layer_4_input_layernorm = (char*)(mpk_provider_ptrs[200]);
  all_tensors["layer_4_input_layernorm"] = layer_4_input_layernorm;
  char *layer_4_qkv_mxfp4 = (char*)(mpk_provider_ptrs[201]);
  all_tensors["layer_4_qkv_mxfp4"] = layer_4_qkv_mxfp4;
  char *layer_4_qkv_bias = (char*)(mpk_provider_ptrs[202]);
  all_tensors["layer_4_qkv_bias"] = layer_4_qkv_bias;
  char *layer_4_k_cache = (char*)(mpk_provider_ptrs[203]);
  all_tensors["layer_4_k_cache"] = layer_4_k_cache;
  char *layer_4_v_cache = (char*)(mpk_provider_ptrs[204]);
  all_tensors["layer_4_v_cache"] = layer_4_v_cache;
  char *layer_4_sinks = (char*)(mpk_provider_ptrs[205]);
  all_tensors["layer_4_sinks"] = layer_4_sinks;
  char *layer_4_o_bias = (char*)(mpk_provider_ptrs[206]);
  all_tensors["layer_4_o_bias"] = layer_4_o_bias;
  char *layer_4_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[207]);
  all_tensors["layer_4_o_proj_mxfp4"] = layer_4_o_proj_mxfp4;
  char *layer_4_post_attn_layernorm = (char*)(mpk_provider_ptrs[208]);
  all_tensors["layer_4_post_attn_layernorm"] = layer_4_post_attn_layernorm;
  char *layer_4_moe_gate = (char*)(mpk_provider_ptrs[209]);
  all_tensors["layer_4_moe_gate"] = layer_4_moe_gate;
  char *layer_4_router_bias = (char*)(mpk_provider_ptrs[210]);
  all_tensors["layer_4_router_bias"] = layer_4_router_bias;
  char *layer_4_gate_up_proj = (char*)(mpk_provider_ptrs[211]);
  all_tensors["layer_4_gate_up_proj"] = layer_4_gate_up_proj;
  char *layer_4_gate_up_bias = (char*)(mpk_provider_ptrs[212]);
  all_tensors["layer_4_gate_up_bias"] = layer_4_gate_up_bias;
  char *layer_4_down_proj = (char*)(mpk_provider_ptrs[213]);
  all_tensors["layer_4_down_proj"] = layer_4_down_proj;
  char *layer_4_down_bias = (char*)(mpk_provider_ptrs[214]);
  all_tensors["layer_4_down_bias"] = layer_4_down_bias;
  char *layer_5_input_layernorm = (char*)(mpk_provider_ptrs[215]);
  all_tensors["layer_5_input_layernorm"] = layer_5_input_layernorm;
  char *layer_5_qkv_mxfp4 = (char*)(mpk_provider_ptrs[216]);
  all_tensors["layer_5_qkv_mxfp4"] = layer_5_qkv_mxfp4;
  char *layer_5_qkv_bias = (char*)(mpk_provider_ptrs[217]);
  all_tensors["layer_5_qkv_bias"] = layer_5_qkv_bias;
  char *layer_5_k_cache = (char*)(mpk_provider_ptrs[218]);
  all_tensors["layer_5_k_cache"] = layer_5_k_cache;
  char *layer_5_v_cache = (char*)(mpk_provider_ptrs[219]);
  all_tensors["layer_5_v_cache"] = layer_5_v_cache;
  char *layer_5_sinks = (char*)(mpk_provider_ptrs[220]);
  all_tensors["layer_5_sinks"] = layer_5_sinks;
  char *layer_5_o_bias = (char*)(mpk_provider_ptrs[221]);
  all_tensors["layer_5_o_bias"] = layer_5_o_bias;
  char *layer_5_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[222]);
  all_tensors["layer_5_o_proj_mxfp4"] = layer_5_o_proj_mxfp4;
  char *layer_5_post_attn_layernorm = (char*)(mpk_provider_ptrs[223]);
  all_tensors["layer_5_post_attn_layernorm"] = layer_5_post_attn_layernorm;
  char *layer_5_moe_gate = (char*)(mpk_provider_ptrs[224]);
  all_tensors["layer_5_moe_gate"] = layer_5_moe_gate;
  char *layer_5_router_bias = (char*)(mpk_provider_ptrs[225]);
  all_tensors["layer_5_router_bias"] = layer_5_router_bias;
  char *layer_5_gate_up_proj = (char*)(mpk_provider_ptrs[226]);
  all_tensors["layer_5_gate_up_proj"] = layer_5_gate_up_proj;
  char *layer_5_gate_up_bias = (char*)(mpk_provider_ptrs[227]);
  all_tensors["layer_5_gate_up_bias"] = layer_5_gate_up_bias;
  char *layer_5_down_proj = (char*)(mpk_provider_ptrs[228]);
  all_tensors["layer_5_down_proj"] = layer_5_down_proj;
  char *layer_5_down_bias = (char*)(mpk_provider_ptrs[229]);
  all_tensors["layer_5_down_bias"] = layer_5_down_bias;
  char *layer_6_input_layernorm = (char*)(mpk_provider_ptrs[230]);
  all_tensors["layer_6_input_layernorm"] = layer_6_input_layernorm;
  char *layer_6_qkv_mxfp4 = (char*)(mpk_provider_ptrs[231]);
  all_tensors["layer_6_qkv_mxfp4"] = layer_6_qkv_mxfp4;
  char *layer_6_qkv_bias = (char*)(mpk_provider_ptrs[232]);
  all_tensors["layer_6_qkv_bias"] = layer_6_qkv_bias;
  char *layer_6_k_cache = (char*)(mpk_provider_ptrs[233]);
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = (char*)(mpk_provider_ptrs[234]);
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_sinks = (char*)(mpk_provider_ptrs[235]);
  all_tensors["layer_6_sinks"] = layer_6_sinks;
  char *layer_6_o_bias = (char*)(mpk_provider_ptrs[236]);
  all_tensors["layer_6_o_bias"] = layer_6_o_bias;
  char *layer_6_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[237]);
  all_tensors["layer_6_o_proj_mxfp4"] = layer_6_o_proj_mxfp4;
  char *layer_6_post_attn_layernorm = (char*)(mpk_provider_ptrs[238]);
  all_tensors["layer_6_post_attn_layernorm"] = layer_6_post_attn_layernorm;
  char *layer_6_moe_gate = (char*)(mpk_provider_ptrs[239]);
  all_tensors["layer_6_moe_gate"] = layer_6_moe_gate;
  char *layer_6_router_bias = (char*)(mpk_provider_ptrs[240]);
  all_tensors["layer_6_router_bias"] = layer_6_router_bias;
  char *layer_6_gate_up_proj = (char*)(mpk_provider_ptrs[241]);
  all_tensors["layer_6_gate_up_proj"] = layer_6_gate_up_proj;
  char *layer_6_gate_up_bias = (char*)(mpk_provider_ptrs[242]);
  all_tensors["layer_6_gate_up_bias"] = layer_6_gate_up_bias;
  char *layer_6_down_proj = (char*)(mpk_provider_ptrs[243]);
  all_tensors["layer_6_down_proj"] = layer_6_down_proj;
  char *layer_6_down_bias = (char*)(mpk_provider_ptrs[244]);
  all_tensors["layer_6_down_bias"] = layer_6_down_bias;
  char *layer_7_input_layernorm = (char*)(mpk_provider_ptrs[245]);
  all_tensors["layer_7_input_layernorm"] = layer_7_input_layernorm;
  char *layer_7_qkv_mxfp4 = (char*)(mpk_provider_ptrs[246]);
  all_tensors["layer_7_qkv_mxfp4"] = layer_7_qkv_mxfp4;
  char *layer_7_qkv_bias = (char*)(mpk_provider_ptrs[247]);
  all_tensors["layer_7_qkv_bias"] = layer_7_qkv_bias;
  char *layer_7_k_cache = (char*)(mpk_provider_ptrs[248]);
  all_tensors["layer_7_k_cache"] = layer_7_k_cache;
  char *layer_7_v_cache = (char*)(mpk_provider_ptrs[249]);
  all_tensors["layer_7_v_cache"] = layer_7_v_cache;
  char *layer_7_sinks = (char*)(mpk_provider_ptrs[250]);
  all_tensors["layer_7_sinks"] = layer_7_sinks;
  char *layer_7_o_bias = (char*)(mpk_provider_ptrs[251]);
  all_tensors["layer_7_o_bias"] = layer_7_o_bias;
  char *layer_7_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[252]);
  all_tensors["layer_7_o_proj_mxfp4"] = layer_7_o_proj_mxfp4;
  char *layer_7_post_attn_layernorm = (char*)(mpk_provider_ptrs[253]);
  all_tensors["layer_7_post_attn_layernorm"] = layer_7_post_attn_layernorm;
  char *layer_7_moe_gate = (char*)(mpk_provider_ptrs[254]);
  all_tensors["layer_7_moe_gate"] = layer_7_moe_gate;
  char *layer_7_router_bias = (char*)(mpk_provider_ptrs[255]);
  all_tensors["layer_7_router_bias"] = layer_7_router_bias;
  char *layer_7_gate_up_proj = (char*)(mpk_provider_ptrs[256]);
  all_tensors["layer_7_gate_up_proj"] = layer_7_gate_up_proj;
  char *layer_7_gate_up_bias = (char*)(mpk_provider_ptrs[257]);
  all_tensors["layer_7_gate_up_bias"] = layer_7_gate_up_bias;
  char *layer_7_down_proj = (char*)(mpk_provider_ptrs[258]);
  all_tensors["layer_7_down_proj"] = layer_7_down_proj;
  char *layer_7_down_bias = (char*)(mpk_provider_ptrs[259]);
  all_tensors["layer_7_down_bias"] = layer_7_down_bias;
  char *layer_8_input_layernorm = (char*)(mpk_provider_ptrs[260]);
  all_tensors["layer_8_input_layernorm"] = layer_8_input_layernorm;
  char *layer_8_qkv_mxfp4 = (char*)(mpk_provider_ptrs[261]);
  all_tensors["layer_8_qkv_mxfp4"] = layer_8_qkv_mxfp4;
  char *layer_8_qkv_bias = (char*)(mpk_provider_ptrs[262]);
  all_tensors["layer_8_qkv_bias"] = layer_8_qkv_bias;
  char *layer_8_k_cache = (char*)(mpk_provider_ptrs[263]);
  all_tensors["layer_8_k_cache"] = layer_8_k_cache;
  char *layer_8_v_cache = (char*)(mpk_provider_ptrs[264]);
  all_tensors["layer_8_v_cache"] = layer_8_v_cache;
  char *layer_8_sinks = (char*)(mpk_provider_ptrs[265]);
  all_tensors["layer_8_sinks"] = layer_8_sinks;
  char *layer_8_o_bias = (char*)(mpk_provider_ptrs[266]);
  all_tensors["layer_8_o_bias"] = layer_8_o_bias;
  char *layer_8_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[267]);
  all_tensors["layer_8_o_proj_mxfp4"] = layer_8_o_proj_mxfp4;
  char *layer_8_post_attn_layernorm = (char*)(mpk_provider_ptrs[268]);
  all_tensors["layer_8_post_attn_layernorm"] = layer_8_post_attn_layernorm;
  char *layer_8_moe_gate = (char*)(mpk_provider_ptrs[269]);
  all_tensors["layer_8_moe_gate"] = layer_8_moe_gate;
  char *layer_8_router_bias = (char*)(mpk_provider_ptrs[270]);
  all_tensors["layer_8_router_bias"] = layer_8_router_bias;
  char *layer_8_gate_up_proj = (char*)(mpk_provider_ptrs[271]);
  all_tensors["layer_8_gate_up_proj"] = layer_8_gate_up_proj;
  char *layer_8_gate_up_bias = (char*)(mpk_provider_ptrs[272]);
  all_tensors["layer_8_gate_up_bias"] = layer_8_gate_up_bias;
  char *layer_8_down_proj = (char*)(mpk_provider_ptrs[273]);
  all_tensors["layer_8_down_proj"] = layer_8_down_proj;
  char *layer_8_down_bias = (char*)(mpk_provider_ptrs[274]);
  all_tensors["layer_8_down_bias"] = layer_8_down_bias;
  char *layer_9_input_layernorm = (char*)(mpk_provider_ptrs[275]);
  all_tensors["layer_9_input_layernorm"] = layer_9_input_layernorm;
  char *layer_9_qkv_mxfp4 = (char*)(mpk_provider_ptrs[276]);
  all_tensors["layer_9_qkv_mxfp4"] = layer_9_qkv_mxfp4;
  char *layer_9_qkv_bias = (char*)(mpk_provider_ptrs[277]);
  all_tensors["layer_9_qkv_bias"] = layer_9_qkv_bias;
  char *layer_9_k_cache = (char*)(mpk_provider_ptrs[278]);
  all_tensors["layer_9_k_cache"] = layer_9_k_cache;
  char *layer_9_v_cache = (char*)(mpk_provider_ptrs[279]);
  all_tensors["layer_9_v_cache"] = layer_9_v_cache;
  char *layer_9_sinks = (char*)(mpk_provider_ptrs[280]);
  all_tensors["layer_9_sinks"] = layer_9_sinks;
  char *layer_9_o_bias = (char*)(mpk_provider_ptrs[281]);
  all_tensors["layer_9_o_bias"] = layer_9_o_bias;
  char *layer_9_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[282]);
  all_tensors["layer_9_o_proj_mxfp4"] = layer_9_o_proj_mxfp4;
  char *layer_9_post_attn_layernorm = (char*)(mpk_provider_ptrs[283]);
  all_tensors["layer_9_post_attn_layernorm"] = layer_9_post_attn_layernorm;
  char *layer_9_moe_gate = (char*)(mpk_provider_ptrs[284]);
  all_tensors["layer_9_moe_gate"] = layer_9_moe_gate;
  char *layer_9_router_bias = (char*)(mpk_provider_ptrs[285]);
  all_tensors["layer_9_router_bias"] = layer_9_router_bias;
  char *layer_9_gate_up_proj = (char*)(mpk_provider_ptrs[286]);
  all_tensors["layer_9_gate_up_proj"] = layer_9_gate_up_proj;
  char *layer_9_gate_up_bias = (char*)(mpk_provider_ptrs[287]);
  all_tensors["layer_9_gate_up_bias"] = layer_9_gate_up_bias;
  char *layer_9_down_proj = (char*)(mpk_provider_ptrs[288]);
  all_tensors["layer_9_down_proj"] = layer_9_down_proj;
  char *layer_9_down_bias = (char*)(mpk_provider_ptrs[289]);
  all_tensors["layer_9_down_bias"] = layer_9_down_bias;
  char *layer_10_input_layernorm = (char*)(mpk_provider_ptrs[290]);
  all_tensors["layer_10_input_layernorm"] = layer_10_input_layernorm;
  char *layer_10_qkv_mxfp4 = (char*)(mpk_provider_ptrs[291]);
  all_tensors["layer_10_qkv_mxfp4"] = layer_10_qkv_mxfp4;
  char *layer_10_qkv_bias = (char*)(mpk_provider_ptrs[292]);
  all_tensors["layer_10_qkv_bias"] = layer_10_qkv_bias;
  char *layer_10_k_cache = (char*)(mpk_provider_ptrs[293]);
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = (char*)(mpk_provider_ptrs[294]);
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_sinks = (char*)(mpk_provider_ptrs[295]);
  all_tensors["layer_10_sinks"] = layer_10_sinks;
  char *layer_10_o_bias = (char*)(mpk_provider_ptrs[296]);
  all_tensors["layer_10_o_bias"] = layer_10_o_bias;
  char *layer_10_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[297]);
  all_tensors["layer_10_o_proj_mxfp4"] = layer_10_o_proj_mxfp4;
  char *layer_10_post_attn_layernorm = (char*)(mpk_provider_ptrs[298]);
  all_tensors["layer_10_post_attn_layernorm"] = layer_10_post_attn_layernorm;
  char *layer_10_moe_gate = (char*)(mpk_provider_ptrs[299]);
  all_tensors["layer_10_moe_gate"] = layer_10_moe_gate;
  char *layer_10_router_bias = (char*)(mpk_provider_ptrs[300]);
  all_tensors["layer_10_router_bias"] = layer_10_router_bias;
  char *layer_10_gate_up_proj = (char*)(mpk_provider_ptrs[301]);
  all_tensors["layer_10_gate_up_proj"] = layer_10_gate_up_proj;
  char *layer_10_gate_up_bias = (char*)(mpk_provider_ptrs[302]);
  all_tensors["layer_10_gate_up_bias"] = layer_10_gate_up_bias;
  char *layer_10_down_proj = (char*)(mpk_provider_ptrs[303]);
  all_tensors["layer_10_down_proj"] = layer_10_down_proj;
  char *layer_10_down_bias = (char*)(mpk_provider_ptrs[304]);
  all_tensors["layer_10_down_bias"] = layer_10_down_bias;
  char *layer_11_input_layernorm = (char*)(mpk_provider_ptrs[305]);
  all_tensors["layer_11_input_layernorm"] = layer_11_input_layernorm;
  char *layer_11_qkv_mxfp4 = (char*)(mpk_provider_ptrs[306]);
  all_tensors["layer_11_qkv_mxfp4"] = layer_11_qkv_mxfp4;
  char *layer_11_qkv_bias = (char*)(mpk_provider_ptrs[307]);
  all_tensors["layer_11_qkv_bias"] = layer_11_qkv_bias;
  char *layer_11_k_cache = (char*)(mpk_provider_ptrs[308]);
  all_tensors["layer_11_k_cache"] = layer_11_k_cache;
  char *layer_11_v_cache = (char*)(mpk_provider_ptrs[309]);
  all_tensors["layer_11_v_cache"] = layer_11_v_cache;
  char *layer_11_sinks = (char*)(mpk_provider_ptrs[310]);
  all_tensors["layer_11_sinks"] = layer_11_sinks;
  char *layer_11_o_bias = (char*)(mpk_provider_ptrs[311]);
  all_tensors["layer_11_o_bias"] = layer_11_o_bias;
  char *layer_11_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[312]);
  all_tensors["layer_11_o_proj_mxfp4"] = layer_11_o_proj_mxfp4;
  char *layer_11_post_attn_layernorm = (char*)(mpk_provider_ptrs[313]);
  all_tensors["layer_11_post_attn_layernorm"] = layer_11_post_attn_layernorm;
  char *layer_11_moe_gate = (char*)(mpk_provider_ptrs[314]);
  all_tensors["layer_11_moe_gate"] = layer_11_moe_gate;
  char *layer_11_router_bias = (char*)(mpk_provider_ptrs[315]);
  all_tensors["layer_11_router_bias"] = layer_11_router_bias;
  char *layer_11_gate_up_proj = (char*)(mpk_provider_ptrs[316]);
  all_tensors["layer_11_gate_up_proj"] = layer_11_gate_up_proj;
  char *layer_11_gate_up_bias = (char*)(mpk_provider_ptrs[317]);
  all_tensors["layer_11_gate_up_bias"] = layer_11_gate_up_bias;
  char *layer_11_down_proj = (char*)(mpk_provider_ptrs[318]);
  all_tensors["layer_11_down_proj"] = layer_11_down_proj;
  char *layer_11_down_bias = (char*)(mpk_provider_ptrs[319]);
  all_tensors["layer_11_down_bias"] = layer_11_down_bias;
  char *layer_12_input_layernorm = (char*)(mpk_provider_ptrs[320]);
  all_tensors["layer_12_input_layernorm"] = layer_12_input_layernorm;
  char *layer_12_qkv_mxfp4 = (char*)(mpk_provider_ptrs[321]);
  all_tensors["layer_12_qkv_mxfp4"] = layer_12_qkv_mxfp4;
  char *layer_12_qkv_bias = (char*)(mpk_provider_ptrs[322]);
  all_tensors["layer_12_qkv_bias"] = layer_12_qkv_bias;
  char *layer_12_k_cache = (char*)(mpk_provider_ptrs[323]);
  all_tensors["layer_12_k_cache"] = layer_12_k_cache;
  char *layer_12_v_cache = (char*)(mpk_provider_ptrs[324]);
  all_tensors["layer_12_v_cache"] = layer_12_v_cache;
  char *layer_12_sinks = (char*)(mpk_provider_ptrs[325]);
  all_tensors["layer_12_sinks"] = layer_12_sinks;
  char *layer_12_o_bias = (char*)(mpk_provider_ptrs[326]);
  all_tensors["layer_12_o_bias"] = layer_12_o_bias;
  char *layer_12_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[327]);
  all_tensors["layer_12_o_proj_mxfp4"] = layer_12_o_proj_mxfp4;
  char *layer_12_post_attn_layernorm = (char*)(mpk_provider_ptrs[328]);
  all_tensors["layer_12_post_attn_layernorm"] = layer_12_post_attn_layernorm;
  char *layer_12_moe_gate = (char*)(mpk_provider_ptrs[329]);
  all_tensors["layer_12_moe_gate"] = layer_12_moe_gate;
  char *layer_12_router_bias = (char*)(mpk_provider_ptrs[330]);
  all_tensors["layer_12_router_bias"] = layer_12_router_bias;
  char *layer_12_gate_up_proj = (char*)(mpk_provider_ptrs[331]);
  all_tensors["layer_12_gate_up_proj"] = layer_12_gate_up_proj;
  char *layer_12_gate_up_bias = (char*)(mpk_provider_ptrs[332]);
  all_tensors["layer_12_gate_up_bias"] = layer_12_gate_up_bias;
  char *layer_12_down_proj = (char*)(mpk_provider_ptrs[333]);
  all_tensors["layer_12_down_proj"] = layer_12_down_proj;
  char *layer_12_down_bias = (char*)(mpk_provider_ptrs[334]);
  all_tensors["layer_12_down_bias"] = layer_12_down_bias;
  char *layer_13_input_layernorm = (char*)(mpk_provider_ptrs[335]);
  all_tensors["layer_13_input_layernorm"] = layer_13_input_layernorm;
  char *layer_13_qkv_mxfp4 = (char*)(mpk_provider_ptrs[336]);
  all_tensors["layer_13_qkv_mxfp4"] = layer_13_qkv_mxfp4;
  char *layer_13_qkv_bias = (char*)(mpk_provider_ptrs[337]);
  all_tensors["layer_13_qkv_bias"] = layer_13_qkv_bias;
  char *layer_13_k_cache = (char*)(mpk_provider_ptrs[338]);
  all_tensors["layer_13_k_cache"] = layer_13_k_cache;
  char *layer_13_v_cache = (char*)(mpk_provider_ptrs[339]);
  all_tensors["layer_13_v_cache"] = layer_13_v_cache;
  char *layer_13_sinks = (char*)(mpk_provider_ptrs[340]);
  all_tensors["layer_13_sinks"] = layer_13_sinks;
  char *layer_13_o_bias = (char*)(mpk_provider_ptrs[341]);
  all_tensors["layer_13_o_bias"] = layer_13_o_bias;
  char *layer_13_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[342]);
  all_tensors["layer_13_o_proj_mxfp4"] = layer_13_o_proj_mxfp4;
  char *layer_13_post_attn_layernorm = (char*)(mpk_provider_ptrs[343]);
  all_tensors["layer_13_post_attn_layernorm"] = layer_13_post_attn_layernorm;
  char *layer_13_moe_gate = (char*)(mpk_provider_ptrs[344]);
  all_tensors["layer_13_moe_gate"] = layer_13_moe_gate;
  char *layer_13_router_bias = (char*)(mpk_provider_ptrs[345]);
  all_tensors["layer_13_router_bias"] = layer_13_router_bias;
  char *layer_13_gate_up_proj = (char*)(mpk_provider_ptrs[346]);
  all_tensors["layer_13_gate_up_proj"] = layer_13_gate_up_proj;
  char *layer_13_gate_up_bias = (char*)(mpk_provider_ptrs[347]);
  all_tensors["layer_13_gate_up_bias"] = layer_13_gate_up_bias;
  char *layer_13_down_proj = (char*)(mpk_provider_ptrs[348]);
  all_tensors["layer_13_down_proj"] = layer_13_down_proj;
  char *layer_13_down_bias = (char*)(mpk_provider_ptrs[349]);
  all_tensors["layer_13_down_bias"] = layer_13_down_bias;
  char *layer_14_input_layernorm = (char*)(mpk_provider_ptrs[350]);
  all_tensors["layer_14_input_layernorm"] = layer_14_input_layernorm;
  char *layer_14_qkv_mxfp4 = (char*)(mpk_provider_ptrs[351]);
  all_tensors["layer_14_qkv_mxfp4"] = layer_14_qkv_mxfp4;
  char *layer_14_qkv_bias = (char*)(mpk_provider_ptrs[352]);
  all_tensors["layer_14_qkv_bias"] = layer_14_qkv_bias;
  char *layer_14_k_cache = (char*)(mpk_provider_ptrs[353]);
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = (char*)(mpk_provider_ptrs[354]);
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_sinks = (char*)(mpk_provider_ptrs[355]);
  all_tensors["layer_14_sinks"] = layer_14_sinks;
  char *layer_14_o_bias = (char*)(mpk_provider_ptrs[356]);
  all_tensors["layer_14_o_bias"] = layer_14_o_bias;
  char *layer_14_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[357]);
  all_tensors["layer_14_o_proj_mxfp4"] = layer_14_o_proj_mxfp4;
  char *layer_14_post_attn_layernorm = (char*)(mpk_provider_ptrs[358]);
  all_tensors["layer_14_post_attn_layernorm"] = layer_14_post_attn_layernorm;
  char *layer_14_moe_gate = (char*)(mpk_provider_ptrs[359]);
  all_tensors["layer_14_moe_gate"] = layer_14_moe_gate;
  char *layer_14_router_bias = (char*)(mpk_provider_ptrs[360]);
  all_tensors["layer_14_router_bias"] = layer_14_router_bias;
  char *layer_14_gate_up_proj = (char*)(mpk_provider_ptrs[361]);
  all_tensors["layer_14_gate_up_proj"] = layer_14_gate_up_proj;
  char *layer_14_gate_up_bias = (char*)(mpk_provider_ptrs[362]);
  all_tensors["layer_14_gate_up_bias"] = layer_14_gate_up_bias;
  char *layer_14_down_proj = (char*)(mpk_provider_ptrs[363]);
  all_tensors["layer_14_down_proj"] = layer_14_down_proj;
  char *layer_14_down_bias = (char*)(mpk_provider_ptrs[364]);
  all_tensors["layer_14_down_bias"] = layer_14_down_bias;
  char *layer_15_input_layernorm = (char*)(mpk_provider_ptrs[365]);
  all_tensors["layer_15_input_layernorm"] = layer_15_input_layernorm;
  char *layer_15_qkv_mxfp4 = (char*)(mpk_provider_ptrs[366]);
  all_tensors["layer_15_qkv_mxfp4"] = layer_15_qkv_mxfp4;
  char *layer_15_qkv_bias = (char*)(mpk_provider_ptrs[367]);
  all_tensors["layer_15_qkv_bias"] = layer_15_qkv_bias;
  char *layer_15_k_cache = (char*)(mpk_provider_ptrs[368]);
  all_tensors["layer_15_k_cache"] = layer_15_k_cache;
  char *layer_15_v_cache = (char*)(mpk_provider_ptrs[369]);
  all_tensors["layer_15_v_cache"] = layer_15_v_cache;
  char *layer_15_sinks = (char*)(mpk_provider_ptrs[370]);
  all_tensors["layer_15_sinks"] = layer_15_sinks;
  char *layer_15_o_bias = (char*)(mpk_provider_ptrs[371]);
  all_tensors["layer_15_o_bias"] = layer_15_o_bias;
  char *layer_15_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[372]);
  all_tensors["layer_15_o_proj_mxfp4"] = layer_15_o_proj_mxfp4;
  char *layer_15_post_attn_layernorm = (char*)(mpk_provider_ptrs[373]);
  all_tensors["layer_15_post_attn_layernorm"] = layer_15_post_attn_layernorm;
  char *layer_15_moe_gate = (char*)(mpk_provider_ptrs[374]);
  all_tensors["layer_15_moe_gate"] = layer_15_moe_gate;
  char *layer_15_router_bias = (char*)(mpk_provider_ptrs[375]);
  all_tensors["layer_15_router_bias"] = layer_15_router_bias;
  char *layer_15_gate_up_proj = (char*)(mpk_provider_ptrs[376]);
  all_tensors["layer_15_gate_up_proj"] = layer_15_gate_up_proj;
  char *layer_15_gate_up_bias = (char*)(mpk_provider_ptrs[377]);
  all_tensors["layer_15_gate_up_bias"] = layer_15_gate_up_bias;
  char *layer_15_down_proj = (char*)(mpk_provider_ptrs[378]);
  all_tensors["layer_15_down_proj"] = layer_15_down_proj;
  char *layer_15_down_bias = (char*)(mpk_provider_ptrs[379]);
  all_tensors["layer_15_down_bias"] = layer_15_down_bias;
  char *layer_16_input_layernorm = (char*)(mpk_provider_ptrs[380]);
  all_tensors["layer_16_input_layernorm"] = layer_16_input_layernorm;
  char *layer_16_qkv_mxfp4 = (char*)(mpk_provider_ptrs[381]);
  all_tensors["layer_16_qkv_mxfp4"] = layer_16_qkv_mxfp4;
  char *layer_16_qkv_bias = (char*)(mpk_provider_ptrs[382]);
  all_tensors["layer_16_qkv_bias"] = layer_16_qkv_bias;
  char *layer_16_k_cache = (char*)(mpk_provider_ptrs[383]);
  all_tensors["layer_16_k_cache"] = layer_16_k_cache;
  char *layer_16_v_cache = (char*)(mpk_provider_ptrs[384]);
  all_tensors["layer_16_v_cache"] = layer_16_v_cache;
  char *layer_16_sinks = (char*)(mpk_provider_ptrs[385]);
  all_tensors["layer_16_sinks"] = layer_16_sinks;
  char *layer_16_o_bias = (char*)(mpk_provider_ptrs[386]);
  all_tensors["layer_16_o_bias"] = layer_16_o_bias;
  char *layer_16_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[387]);
  all_tensors["layer_16_o_proj_mxfp4"] = layer_16_o_proj_mxfp4;
  char *layer_16_post_attn_layernorm = (char*)(mpk_provider_ptrs[388]);
  all_tensors["layer_16_post_attn_layernorm"] = layer_16_post_attn_layernorm;
  char *layer_16_moe_gate = (char*)(mpk_provider_ptrs[389]);
  all_tensors["layer_16_moe_gate"] = layer_16_moe_gate;
  char *layer_16_router_bias = (char*)(mpk_provider_ptrs[390]);
  all_tensors["layer_16_router_bias"] = layer_16_router_bias;
  char *layer_16_gate_up_proj = (char*)(mpk_provider_ptrs[391]);
  all_tensors["layer_16_gate_up_proj"] = layer_16_gate_up_proj;
  char *layer_16_gate_up_bias = (char*)(mpk_provider_ptrs[392]);
  all_tensors["layer_16_gate_up_bias"] = layer_16_gate_up_bias;
  char *layer_16_down_proj = (char*)(mpk_provider_ptrs[393]);
  all_tensors["layer_16_down_proj"] = layer_16_down_proj;
  char *layer_16_down_bias = (char*)(mpk_provider_ptrs[394]);
  all_tensors["layer_16_down_bias"] = layer_16_down_bias;
  char *layer_17_input_layernorm = (char*)(mpk_provider_ptrs[395]);
  all_tensors["layer_17_input_layernorm"] = layer_17_input_layernorm;
  char *layer_17_qkv_mxfp4 = (char*)(mpk_provider_ptrs[396]);
  all_tensors["layer_17_qkv_mxfp4"] = layer_17_qkv_mxfp4;
  char *layer_17_qkv_bias = (char*)(mpk_provider_ptrs[397]);
  all_tensors["layer_17_qkv_bias"] = layer_17_qkv_bias;
  char *layer_17_k_cache = (char*)(mpk_provider_ptrs[398]);
  all_tensors["layer_17_k_cache"] = layer_17_k_cache;
  char *layer_17_v_cache = (char*)(mpk_provider_ptrs[399]);
  all_tensors["layer_17_v_cache"] = layer_17_v_cache;
  char *layer_17_sinks = (char*)(mpk_provider_ptrs[400]);
  all_tensors["layer_17_sinks"] = layer_17_sinks;
  char *layer_17_o_bias = (char*)(mpk_provider_ptrs[401]);
  all_tensors["layer_17_o_bias"] = layer_17_o_bias;
  char *layer_17_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[402]);
  all_tensors["layer_17_o_proj_mxfp4"] = layer_17_o_proj_mxfp4;
  char *layer_17_post_attn_layernorm = (char*)(mpk_provider_ptrs[403]);
  all_tensors["layer_17_post_attn_layernorm"] = layer_17_post_attn_layernorm;
  char *layer_17_moe_gate = (char*)(mpk_provider_ptrs[404]);
  all_tensors["layer_17_moe_gate"] = layer_17_moe_gate;
  char *layer_17_router_bias = (char*)(mpk_provider_ptrs[405]);
  all_tensors["layer_17_router_bias"] = layer_17_router_bias;
  char *layer_17_gate_up_proj = (char*)(mpk_provider_ptrs[406]);
  all_tensors["layer_17_gate_up_proj"] = layer_17_gate_up_proj;
  char *layer_17_gate_up_bias = (char*)(mpk_provider_ptrs[407]);
  all_tensors["layer_17_gate_up_bias"] = layer_17_gate_up_bias;
  char *layer_17_down_proj = (char*)(mpk_provider_ptrs[408]);
  all_tensors["layer_17_down_proj"] = layer_17_down_proj;
  char *layer_17_down_bias = (char*)(mpk_provider_ptrs[409]);
  all_tensors["layer_17_down_bias"] = layer_17_down_bias;
  char *layer_18_input_layernorm = (char*)(mpk_provider_ptrs[410]);
  all_tensors["layer_18_input_layernorm"] = layer_18_input_layernorm;
  char *layer_18_qkv_mxfp4 = (char*)(mpk_provider_ptrs[411]);
  all_tensors["layer_18_qkv_mxfp4"] = layer_18_qkv_mxfp4;
  char *layer_18_qkv_bias = (char*)(mpk_provider_ptrs[412]);
  all_tensors["layer_18_qkv_bias"] = layer_18_qkv_bias;
  char *layer_18_k_cache = (char*)(mpk_provider_ptrs[413]);
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = (char*)(mpk_provider_ptrs[414]);
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_sinks = (char*)(mpk_provider_ptrs[415]);
  all_tensors["layer_18_sinks"] = layer_18_sinks;
  char *layer_18_o_bias = (char*)(mpk_provider_ptrs[416]);
  all_tensors["layer_18_o_bias"] = layer_18_o_bias;
  char *layer_18_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[417]);
  all_tensors["layer_18_o_proj_mxfp4"] = layer_18_o_proj_mxfp4;
  char *layer_18_post_attn_layernorm = (char*)(mpk_provider_ptrs[418]);
  all_tensors["layer_18_post_attn_layernorm"] = layer_18_post_attn_layernorm;
  char *layer_18_moe_gate = (char*)(mpk_provider_ptrs[419]);
  all_tensors["layer_18_moe_gate"] = layer_18_moe_gate;
  char *layer_18_router_bias = (char*)(mpk_provider_ptrs[420]);
  all_tensors["layer_18_router_bias"] = layer_18_router_bias;
  char *layer_18_gate_up_proj = (char*)(mpk_provider_ptrs[421]);
  all_tensors["layer_18_gate_up_proj"] = layer_18_gate_up_proj;
  char *layer_18_gate_up_bias = (char*)(mpk_provider_ptrs[422]);
  all_tensors["layer_18_gate_up_bias"] = layer_18_gate_up_bias;
  char *layer_18_down_proj = (char*)(mpk_provider_ptrs[423]);
  all_tensors["layer_18_down_proj"] = layer_18_down_proj;
  char *layer_18_down_bias = (char*)(mpk_provider_ptrs[424]);
  all_tensors["layer_18_down_bias"] = layer_18_down_bias;
  char *layer_19_input_layernorm = (char*)(mpk_provider_ptrs[425]);
  all_tensors["layer_19_input_layernorm"] = layer_19_input_layernorm;
  char *layer_19_qkv_mxfp4 = (char*)(mpk_provider_ptrs[426]);
  all_tensors["layer_19_qkv_mxfp4"] = layer_19_qkv_mxfp4;
  char *layer_19_qkv_bias = (char*)(mpk_provider_ptrs[427]);
  all_tensors["layer_19_qkv_bias"] = layer_19_qkv_bias;
  char *layer_19_k_cache = (char*)(mpk_provider_ptrs[428]);
  all_tensors["layer_19_k_cache"] = layer_19_k_cache;
  char *layer_19_v_cache = (char*)(mpk_provider_ptrs[429]);
  all_tensors["layer_19_v_cache"] = layer_19_v_cache;
  char *layer_19_sinks = (char*)(mpk_provider_ptrs[430]);
  all_tensors["layer_19_sinks"] = layer_19_sinks;
  char *layer_19_o_bias = (char*)(mpk_provider_ptrs[431]);
  all_tensors["layer_19_o_bias"] = layer_19_o_bias;
  char *layer_19_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[432]);
  all_tensors["layer_19_o_proj_mxfp4"] = layer_19_o_proj_mxfp4;
  char *layer_19_post_attn_layernorm = (char*)(mpk_provider_ptrs[433]);
  all_tensors["layer_19_post_attn_layernorm"] = layer_19_post_attn_layernorm;
  char *layer_19_moe_gate = (char*)(mpk_provider_ptrs[434]);
  all_tensors["layer_19_moe_gate"] = layer_19_moe_gate;
  char *layer_19_router_bias = (char*)(mpk_provider_ptrs[435]);
  all_tensors["layer_19_router_bias"] = layer_19_router_bias;
  char *layer_19_gate_up_proj = (char*)(mpk_provider_ptrs[436]);
  all_tensors["layer_19_gate_up_proj"] = layer_19_gate_up_proj;
  char *layer_19_gate_up_bias = (char*)(mpk_provider_ptrs[437]);
  all_tensors["layer_19_gate_up_bias"] = layer_19_gate_up_bias;
  char *layer_19_down_proj = (char*)(mpk_provider_ptrs[438]);
  all_tensors["layer_19_down_proj"] = layer_19_down_proj;
  char *layer_19_down_bias = (char*)(mpk_provider_ptrs[439]);
  all_tensors["layer_19_down_bias"] = layer_19_down_bias;
  char *layer_20_input_layernorm = (char*)(mpk_provider_ptrs[440]);
  all_tensors["layer_20_input_layernorm"] = layer_20_input_layernorm;
  char *layer_20_qkv_mxfp4 = (char*)(mpk_provider_ptrs[441]);
  all_tensors["layer_20_qkv_mxfp4"] = layer_20_qkv_mxfp4;
  char *layer_20_qkv_bias = (char*)(mpk_provider_ptrs[442]);
  all_tensors["layer_20_qkv_bias"] = layer_20_qkv_bias;
  char *layer_20_k_cache = (char*)(mpk_provider_ptrs[443]);
  all_tensors["layer_20_k_cache"] = layer_20_k_cache;
  char *layer_20_v_cache = (char*)(mpk_provider_ptrs[444]);
  all_tensors["layer_20_v_cache"] = layer_20_v_cache;
  char *layer_20_sinks = (char*)(mpk_provider_ptrs[445]);
  all_tensors["layer_20_sinks"] = layer_20_sinks;
  char *layer_20_o_bias = (char*)(mpk_provider_ptrs[446]);
  all_tensors["layer_20_o_bias"] = layer_20_o_bias;
  char *layer_20_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[447]);
  all_tensors["layer_20_o_proj_mxfp4"] = layer_20_o_proj_mxfp4;
  char *layer_20_post_attn_layernorm = (char*)(mpk_provider_ptrs[448]);
  all_tensors["layer_20_post_attn_layernorm"] = layer_20_post_attn_layernorm;
  char *layer_20_moe_gate = (char*)(mpk_provider_ptrs[449]);
  all_tensors["layer_20_moe_gate"] = layer_20_moe_gate;
  char *layer_20_router_bias = (char*)(mpk_provider_ptrs[450]);
  all_tensors["layer_20_router_bias"] = layer_20_router_bias;
  char *layer_20_gate_up_proj = (char*)(mpk_provider_ptrs[451]);
  all_tensors["layer_20_gate_up_proj"] = layer_20_gate_up_proj;
  char *layer_20_gate_up_bias = (char*)(mpk_provider_ptrs[452]);
  all_tensors["layer_20_gate_up_bias"] = layer_20_gate_up_bias;
  char *layer_20_down_proj = (char*)(mpk_provider_ptrs[453]);
  all_tensors["layer_20_down_proj"] = layer_20_down_proj;
  char *layer_20_down_bias = (char*)(mpk_provider_ptrs[454]);
  all_tensors["layer_20_down_bias"] = layer_20_down_bias;
  char *layer_21_input_layernorm = (char*)(mpk_provider_ptrs[455]);
  all_tensors["layer_21_input_layernorm"] = layer_21_input_layernorm;
  char *layer_21_qkv_mxfp4 = (char*)(mpk_provider_ptrs[456]);
  all_tensors["layer_21_qkv_mxfp4"] = layer_21_qkv_mxfp4;
  char *layer_21_qkv_bias = (char*)(mpk_provider_ptrs[457]);
  all_tensors["layer_21_qkv_bias"] = layer_21_qkv_bias;
  char *layer_21_k_cache = (char*)(mpk_provider_ptrs[458]);
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = (char*)(mpk_provider_ptrs[459]);
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_sinks = (char*)(mpk_provider_ptrs[460]);
  all_tensors["layer_21_sinks"] = layer_21_sinks;
  char *layer_21_o_bias = (char*)(mpk_provider_ptrs[461]);
  all_tensors["layer_21_o_bias"] = layer_21_o_bias;
  char *layer_21_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[462]);
  all_tensors["layer_21_o_proj_mxfp4"] = layer_21_o_proj_mxfp4;
  char *layer_21_post_attn_layernorm = (char*)(mpk_provider_ptrs[463]);
  all_tensors["layer_21_post_attn_layernorm"] = layer_21_post_attn_layernorm;
  char *layer_21_moe_gate = (char*)(mpk_provider_ptrs[464]);
  all_tensors["layer_21_moe_gate"] = layer_21_moe_gate;
  char *layer_21_router_bias = (char*)(mpk_provider_ptrs[465]);
  all_tensors["layer_21_router_bias"] = layer_21_router_bias;
  char *layer_21_gate_up_proj = (char*)(mpk_provider_ptrs[466]);
  all_tensors["layer_21_gate_up_proj"] = layer_21_gate_up_proj;
  char *layer_21_gate_up_bias = (char*)(mpk_provider_ptrs[467]);
  all_tensors["layer_21_gate_up_bias"] = layer_21_gate_up_bias;
  char *layer_21_down_proj = (char*)(mpk_provider_ptrs[468]);
  all_tensors["layer_21_down_proj"] = layer_21_down_proj;
  char *layer_21_down_bias = (char*)(mpk_provider_ptrs[469]);
  all_tensors["layer_21_down_bias"] = layer_21_down_bias;
  char *layer_22_input_layernorm = (char*)(mpk_provider_ptrs[470]);
  all_tensors["layer_22_input_layernorm"] = layer_22_input_layernorm;
  char *layer_22_qkv_mxfp4 = (char*)(mpk_provider_ptrs[471]);
  all_tensors["layer_22_qkv_mxfp4"] = layer_22_qkv_mxfp4;
  char *layer_22_qkv_bias = (char*)(mpk_provider_ptrs[472]);
  all_tensors["layer_22_qkv_bias"] = layer_22_qkv_bias;
  char *layer_22_k_cache = (char*)(mpk_provider_ptrs[473]);
  all_tensors["layer_22_k_cache"] = layer_22_k_cache;
  char *layer_22_v_cache = (char*)(mpk_provider_ptrs[474]);
  all_tensors["layer_22_v_cache"] = layer_22_v_cache;
  char *layer_22_sinks = (char*)(mpk_provider_ptrs[475]);
  all_tensors["layer_22_sinks"] = layer_22_sinks;
  char *layer_22_o_bias = (char*)(mpk_provider_ptrs[476]);
  all_tensors["layer_22_o_bias"] = layer_22_o_bias;
  char *layer_22_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[477]);
  all_tensors["layer_22_o_proj_mxfp4"] = layer_22_o_proj_mxfp4;
  char *layer_22_post_attn_layernorm = (char*)(mpk_provider_ptrs[478]);
  all_tensors["layer_22_post_attn_layernorm"] = layer_22_post_attn_layernorm;
  char *layer_22_moe_gate = (char*)(mpk_provider_ptrs[479]);
  all_tensors["layer_22_moe_gate"] = layer_22_moe_gate;
  char *layer_22_router_bias = (char*)(mpk_provider_ptrs[480]);
  all_tensors["layer_22_router_bias"] = layer_22_router_bias;
  char *layer_22_gate_up_proj = (char*)(mpk_provider_ptrs[481]);
  all_tensors["layer_22_gate_up_proj"] = layer_22_gate_up_proj;
  char *layer_22_gate_up_bias = (char*)(mpk_provider_ptrs[482]);
  all_tensors["layer_22_gate_up_bias"] = layer_22_gate_up_bias;
  char *layer_22_down_proj = (char*)(mpk_provider_ptrs[483]);
  all_tensors["layer_22_down_proj"] = layer_22_down_proj;
  char *layer_22_down_bias = (char*)(mpk_provider_ptrs[484]);
  all_tensors["layer_22_down_bias"] = layer_22_down_bias;
  char *layer_23_input_layernorm = (char*)(mpk_provider_ptrs[485]);
  all_tensors["layer_23_input_layernorm"] = layer_23_input_layernorm;
  char *layer_23_qkv_mxfp4 = (char*)(mpk_provider_ptrs[486]);
  all_tensors["layer_23_qkv_mxfp4"] = layer_23_qkv_mxfp4;
  char *layer_23_qkv_bias = (char*)(mpk_provider_ptrs[487]);
  all_tensors["layer_23_qkv_bias"] = layer_23_qkv_bias;
  char *layer_23_k_cache = (char*)(mpk_provider_ptrs[488]);
  all_tensors["layer_23_k_cache"] = layer_23_k_cache;
  char *layer_23_v_cache = (char*)(mpk_provider_ptrs[489]);
  all_tensors["layer_23_v_cache"] = layer_23_v_cache;
  char *layer_23_sinks = (char*)(mpk_provider_ptrs[490]);
  all_tensors["layer_23_sinks"] = layer_23_sinks;
  char *layer_23_o_bias = (char*)(mpk_provider_ptrs[491]);
  all_tensors["layer_23_o_bias"] = layer_23_o_bias;
  char *layer_23_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[492]);
  all_tensors["layer_23_o_proj_mxfp4"] = layer_23_o_proj_mxfp4;
  char *layer_23_post_attn_layernorm = (char*)(mpk_provider_ptrs[493]);
  all_tensors["layer_23_post_attn_layernorm"] = layer_23_post_attn_layernorm;
  char *layer_23_moe_gate = (char*)(mpk_provider_ptrs[494]);
  all_tensors["layer_23_moe_gate"] = layer_23_moe_gate;
  char *layer_23_router_bias = (char*)(mpk_provider_ptrs[495]);
  all_tensors["layer_23_router_bias"] = layer_23_router_bias;
  char *layer_23_gate_up_proj = (char*)(mpk_provider_ptrs[496]);
  all_tensors["layer_23_gate_up_proj"] = layer_23_gate_up_proj;
  char *layer_23_gate_up_bias = (char*)(mpk_provider_ptrs[497]);
  all_tensors["layer_23_gate_up_bias"] = layer_23_gate_up_bias;
  char *layer_23_down_proj = (char*)(mpk_provider_ptrs[498]);
  all_tensors["layer_23_down_proj"] = layer_23_down_proj;
  char *layer_23_down_bias = (char*)(mpk_provider_ptrs[499]);
  all_tensors["layer_23_down_bias"] = layer_23_down_bias;
  char *layer_24_input_layernorm = (char*)(mpk_provider_ptrs[500]);
  all_tensors["layer_24_input_layernorm"] = layer_24_input_layernorm;
  char *layer_24_qkv_mxfp4 = (char*)(mpk_provider_ptrs[501]);
  all_tensors["layer_24_qkv_mxfp4"] = layer_24_qkv_mxfp4;
  char *layer_24_qkv_bias = (char*)(mpk_provider_ptrs[502]);
  all_tensors["layer_24_qkv_bias"] = layer_24_qkv_bias;
  char *layer_24_k_cache = (char*)(mpk_provider_ptrs[503]);
  all_tensors["layer_24_k_cache"] = layer_24_k_cache;
  char *layer_24_v_cache = (char*)(mpk_provider_ptrs[504]);
  all_tensors["layer_24_v_cache"] = layer_24_v_cache;
  char *layer_24_sinks = (char*)(mpk_provider_ptrs[505]);
  all_tensors["layer_24_sinks"] = layer_24_sinks;
  char *layer_24_o_bias = (char*)(mpk_provider_ptrs[506]);
  all_tensors["layer_24_o_bias"] = layer_24_o_bias;
  char *layer_24_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[507]);
  all_tensors["layer_24_o_proj_mxfp4"] = layer_24_o_proj_mxfp4;
  char *layer_24_post_attn_layernorm = (char*)(mpk_provider_ptrs[508]);
  all_tensors["layer_24_post_attn_layernorm"] = layer_24_post_attn_layernorm;
  char *layer_24_moe_gate = (char*)(mpk_provider_ptrs[509]);
  all_tensors["layer_24_moe_gate"] = layer_24_moe_gate;
  char *layer_24_router_bias = (char*)(mpk_provider_ptrs[510]);
  all_tensors["layer_24_router_bias"] = layer_24_router_bias;
  char *layer_24_gate_up_proj = (char*)(mpk_provider_ptrs[511]);
  all_tensors["layer_24_gate_up_proj"] = layer_24_gate_up_proj;
  char *layer_24_gate_up_bias = (char*)(mpk_provider_ptrs[512]);
  all_tensors["layer_24_gate_up_bias"] = layer_24_gate_up_bias;
  char *layer_24_down_proj = (char*)(mpk_provider_ptrs[513]);
  all_tensors["layer_24_down_proj"] = layer_24_down_proj;
  char *layer_24_down_bias = (char*)(mpk_provider_ptrs[514]);
  all_tensors["layer_24_down_bias"] = layer_24_down_bias;
  char *layer_25_input_layernorm = (char*)(mpk_provider_ptrs[515]);
  all_tensors["layer_25_input_layernorm"] = layer_25_input_layernorm;
  char *layer_25_qkv_mxfp4 = (char*)(mpk_provider_ptrs[516]);
  all_tensors["layer_25_qkv_mxfp4"] = layer_25_qkv_mxfp4;
  char *layer_25_qkv_bias = (char*)(mpk_provider_ptrs[517]);
  all_tensors["layer_25_qkv_bias"] = layer_25_qkv_bias;
  char *layer_25_k_cache = (char*)(mpk_provider_ptrs[518]);
  all_tensors["layer_25_k_cache"] = layer_25_k_cache;
  char *layer_25_v_cache = (char*)(mpk_provider_ptrs[519]);
  all_tensors["layer_25_v_cache"] = layer_25_v_cache;
  char *layer_25_sinks = (char*)(mpk_provider_ptrs[520]);
  all_tensors["layer_25_sinks"] = layer_25_sinks;
  char *layer_25_o_bias = (char*)(mpk_provider_ptrs[521]);
  all_tensors["layer_25_o_bias"] = layer_25_o_bias;
  char *layer_25_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[522]);
  all_tensors["layer_25_o_proj_mxfp4"] = layer_25_o_proj_mxfp4;
  char *layer_25_post_attn_layernorm = (char*)(mpk_provider_ptrs[523]);
  all_tensors["layer_25_post_attn_layernorm"] = layer_25_post_attn_layernorm;
  char *layer_25_moe_gate = (char*)(mpk_provider_ptrs[524]);
  all_tensors["layer_25_moe_gate"] = layer_25_moe_gate;
  char *layer_25_router_bias = (char*)(mpk_provider_ptrs[525]);
  all_tensors["layer_25_router_bias"] = layer_25_router_bias;
  char *layer_25_gate_up_proj = (char*)(mpk_provider_ptrs[526]);
  all_tensors["layer_25_gate_up_proj"] = layer_25_gate_up_proj;
  char *layer_25_gate_up_bias = (char*)(mpk_provider_ptrs[527]);
  all_tensors["layer_25_gate_up_bias"] = layer_25_gate_up_bias;
  char *layer_25_down_proj = (char*)(mpk_provider_ptrs[528]);
  all_tensors["layer_25_down_proj"] = layer_25_down_proj;
  char *layer_25_down_bias = (char*)(mpk_provider_ptrs[529]);
  all_tensors["layer_25_down_bias"] = layer_25_down_bias;
  char *layer_26_input_layernorm = (char*)(mpk_provider_ptrs[530]);
  all_tensors["layer_26_input_layernorm"] = layer_26_input_layernorm;
  char *layer_26_qkv_mxfp4 = (char*)(mpk_provider_ptrs[531]);
  all_tensors["layer_26_qkv_mxfp4"] = layer_26_qkv_mxfp4;
  char *layer_26_qkv_bias = (char*)(mpk_provider_ptrs[532]);
  all_tensors["layer_26_qkv_bias"] = layer_26_qkv_bias;
  char *layer_26_k_cache = (char*)(mpk_provider_ptrs[533]);
  all_tensors["layer_26_k_cache"] = layer_26_k_cache;
  char *layer_26_v_cache = (char*)(mpk_provider_ptrs[534]);
  all_tensors["layer_26_v_cache"] = layer_26_v_cache;
  char *layer_26_sinks = (char*)(mpk_provider_ptrs[535]);
  all_tensors["layer_26_sinks"] = layer_26_sinks;
  char *layer_26_o_bias = (char*)(mpk_provider_ptrs[536]);
  all_tensors["layer_26_o_bias"] = layer_26_o_bias;
  char *layer_26_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[537]);
  all_tensors["layer_26_o_proj_mxfp4"] = layer_26_o_proj_mxfp4;
  char *layer_26_post_attn_layernorm = (char*)(mpk_provider_ptrs[538]);
  all_tensors["layer_26_post_attn_layernorm"] = layer_26_post_attn_layernorm;
  char *layer_26_moe_gate = (char*)(mpk_provider_ptrs[539]);
  all_tensors["layer_26_moe_gate"] = layer_26_moe_gate;
  char *layer_26_router_bias = (char*)(mpk_provider_ptrs[540]);
  all_tensors["layer_26_router_bias"] = layer_26_router_bias;
  char *layer_26_gate_up_proj = (char*)(mpk_provider_ptrs[541]);
  all_tensors["layer_26_gate_up_proj"] = layer_26_gate_up_proj;
  char *layer_26_gate_up_bias = (char*)(mpk_provider_ptrs[542]);
  all_tensors["layer_26_gate_up_bias"] = layer_26_gate_up_bias;
  char *layer_26_down_proj = (char*)(mpk_provider_ptrs[543]);
  all_tensors["layer_26_down_proj"] = layer_26_down_proj;
  char *layer_26_down_bias = (char*)(mpk_provider_ptrs[544]);
  all_tensors["layer_26_down_bias"] = layer_26_down_bias;
  char *layer_27_input_layernorm = (char*)(mpk_provider_ptrs[545]);
  all_tensors["layer_27_input_layernorm"] = layer_27_input_layernorm;
  char *layer_27_qkv_mxfp4 = (char*)(mpk_provider_ptrs[546]);
  all_tensors["layer_27_qkv_mxfp4"] = layer_27_qkv_mxfp4;
  char *layer_27_qkv_bias = (char*)(mpk_provider_ptrs[547]);
  all_tensors["layer_27_qkv_bias"] = layer_27_qkv_bias;
  char *layer_27_k_cache = (char*)(mpk_provider_ptrs[548]);
  all_tensors["layer_27_k_cache"] = layer_27_k_cache;
  char *layer_27_v_cache = (char*)(mpk_provider_ptrs[549]);
  all_tensors["layer_27_v_cache"] = layer_27_v_cache;
  char *layer_27_sinks = (char*)(mpk_provider_ptrs[550]);
  all_tensors["layer_27_sinks"] = layer_27_sinks;
  char *layer_27_o_bias = (char*)(mpk_provider_ptrs[551]);
  all_tensors["layer_27_o_bias"] = layer_27_o_bias;
  char *layer_27_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[552]);
  all_tensors["layer_27_o_proj_mxfp4"] = layer_27_o_proj_mxfp4;
  char *layer_27_post_attn_layernorm = (char*)(mpk_provider_ptrs[553]);
  all_tensors["layer_27_post_attn_layernorm"] = layer_27_post_attn_layernorm;
  char *layer_27_moe_gate = (char*)(mpk_provider_ptrs[554]);
  all_tensors["layer_27_moe_gate"] = layer_27_moe_gate;
  char *layer_27_router_bias = (char*)(mpk_provider_ptrs[555]);
  all_tensors["layer_27_router_bias"] = layer_27_router_bias;
  char *layer_27_gate_up_proj = (char*)(mpk_provider_ptrs[556]);
  all_tensors["layer_27_gate_up_proj"] = layer_27_gate_up_proj;
  char *layer_27_gate_up_bias = (char*)(mpk_provider_ptrs[557]);
  all_tensors["layer_27_gate_up_bias"] = layer_27_gate_up_bias;
  char *layer_27_down_proj = (char*)(mpk_provider_ptrs[558]);
  all_tensors["layer_27_down_proj"] = layer_27_down_proj;
  char *layer_27_down_bias = (char*)(mpk_provider_ptrs[559]);
  all_tensors["layer_27_down_bias"] = layer_27_down_bias;
  char *layer_28_input_layernorm = (char*)(mpk_provider_ptrs[560]);
  all_tensors["layer_28_input_layernorm"] = layer_28_input_layernorm;
  char *layer_28_qkv_mxfp4 = (char*)(mpk_provider_ptrs[561]);
  all_tensors["layer_28_qkv_mxfp4"] = layer_28_qkv_mxfp4;
  char *layer_28_qkv_bias = (char*)(mpk_provider_ptrs[562]);
  all_tensors["layer_28_qkv_bias"] = layer_28_qkv_bias;
  char *layer_28_k_cache = (char*)(mpk_provider_ptrs[563]);
  all_tensors["layer_28_k_cache"] = layer_28_k_cache;
  char *layer_28_v_cache = (char*)(mpk_provider_ptrs[564]);
  all_tensors["layer_28_v_cache"] = layer_28_v_cache;
  char *layer_28_sinks = (char*)(mpk_provider_ptrs[565]);
  all_tensors["layer_28_sinks"] = layer_28_sinks;
  char *layer_28_o_bias = (char*)(mpk_provider_ptrs[566]);
  all_tensors["layer_28_o_bias"] = layer_28_o_bias;
  char *layer_28_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[567]);
  all_tensors["layer_28_o_proj_mxfp4"] = layer_28_o_proj_mxfp4;
  char *layer_28_post_attn_layernorm = (char*)(mpk_provider_ptrs[568]);
  all_tensors["layer_28_post_attn_layernorm"] = layer_28_post_attn_layernorm;
  char *layer_28_moe_gate = (char*)(mpk_provider_ptrs[569]);
  all_tensors["layer_28_moe_gate"] = layer_28_moe_gate;
  char *layer_28_router_bias = (char*)(mpk_provider_ptrs[570]);
  all_tensors["layer_28_router_bias"] = layer_28_router_bias;
  char *layer_28_gate_up_proj = (char*)(mpk_provider_ptrs[571]);
  all_tensors["layer_28_gate_up_proj"] = layer_28_gate_up_proj;
  char *layer_28_gate_up_bias = (char*)(mpk_provider_ptrs[572]);
  all_tensors["layer_28_gate_up_bias"] = layer_28_gate_up_bias;
  char *layer_28_down_proj = (char*)(mpk_provider_ptrs[573]);
  all_tensors["layer_28_down_proj"] = layer_28_down_proj;
  char *layer_28_down_bias = (char*)(mpk_provider_ptrs[574]);
  all_tensors["layer_28_down_bias"] = layer_28_down_bias;
  char *layer_29_input_layernorm = (char*)(mpk_provider_ptrs[575]);
  all_tensors["layer_29_input_layernorm"] = layer_29_input_layernorm;
  char *layer_29_qkv_mxfp4 = (char*)(mpk_provider_ptrs[576]);
  all_tensors["layer_29_qkv_mxfp4"] = layer_29_qkv_mxfp4;
  char *layer_29_qkv_bias = (char*)(mpk_provider_ptrs[577]);
  all_tensors["layer_29_qkv_bias"] = layer_29_qkv_bias;
  char *layer_29_k_cache = (char*)(mpk_provider_ptrs[578]);
  all_tensors["layer_29_k_cache"] = layer_29_k_cache;
  char *layer_29_v_cache = (char*)(mpk_provider_ptrs[579]);
  all_tensors["layer_29_v_cache"] = layer_29_v_cache;
  char *layer_29_sinks = (char*)(mpk_provider_ptrs[580]);
  all_tensors["layer_29_sinks"] = layer_29_sinks;
  char *layer_29_o_bias = (char*)(mpk_provider_ptrs[581]);
  all_tensors["layer_29_o_bias"] = layer_29_o_bias;
  char *layer_29_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[582]);
  all_tensors["layer_29_o_proj_mxfp4"] = layer_29_o_proj_mxfp4;
  char *layer_29_post_attn_layernorm = (char*)(mpk_provider_ptrs[583]);
  all_tensors["layer_29_post_attn_layernorm"] = layer_29_post_attn_layernorm;
  char *layer_29_moe_gate = (char*)(mpk_provider_ptrs[584]);
  all_tensors["layer_29_moe_gate"] = layer_29_moe_gate;
  char *layer_29_router_bias = (char*)(mpk_provider_ptrs[585]);
  all_tensors["layer_29_router_bias"] = layer_29_router_bias;
  char *layer_29_gate_up_proj = (char*)(mpk_provider_ptrs[586]);
  all_tensors["layer_29_gate_up_proj"] = layer_29_gate_up_proj;
  char *layer_29_gate_up_bias = (char*)(mpk_provider_ptrs[587]);
  all_tensors["layer_29_gate_up_bias"] = layer_29_gate_up_bias;
  char *layer_29_down_proj = (char*)(mpk_provider_ptrs[588]);
  all_tensors["layer_29_down_proj"] = layer_29_down_proj;
  char *layer_29_down_bias = (char*)(mpk_provider_ptrs[589]);
  all_tensors["layer_29_down_bias"] = layer_29_down_bias;
  char *layer_30_input_layernorm = (char*)(mpk_provider_ptrs[590]);
  all_tensors["layer_30_input_layernorm"] = layer_30_input_layernorm;
  char *layer_30_qkv_mxfp4 = (char*)(mpk_provider_ptrs[591]);
  all_tensors["layer_30_qkv_mxfp4"] = layer_30_qkv_mxfp4;
  char *layer_30_qkv_bias = (char*)(mpk_provider_ptrs[592]);
  all_tensors["layer_30_qkv_bias"] = layer_30_qkv_bias;
  char *layer_30_k_cache = (char*)(mpk_provider_ptrs[593]);
  all_tensors["layer_30_k_cache"] = layer_30_k_cache;
  char *layer_30_v_cache = (char*)(mpk_provider_ptrs[594]);
  all_tensors["layer_30_v_cache"] = layer_30_v_cache;
  char *layer_30_sinks = (char*)(mpk_provider_ptrs[595]);
  all_tensors["layer_30_sinks"] = layer_30_sinks;
  char *layer_30_o_bias = (char*)(mpk_provider_ptrs[596]);
  all_tensors["layer_30_o_bias"] = layer_30_o_bias;
  char *layer_30_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[597]);
  all_tensors["layer_30_o_proj_mxfp4"] = layer_30_o_proj_mxfp4;
  char *layer_30_post_attn_layernorm = (char*)(mpk_provider_ptrs[598]);
  all_tensors["layer_30_post_attn_layernorm"] = layer_30_post_attn_layernorm;
  char *layer_30_moe_gate = (char*)(mpk_provider_ptrs[599]);
  all_tensors["layer_30_moe_gate"] = layer_30_moe_gate;
  char *layer_30_router_bias = (char*)(mpk_provider_ptrs[600]);
  all_tensors["layer_30_router_bias"] = layer_30_router_bias;
  char *layer_30_gate_up_proj = (char*)(mpk_provider_ptrs[601]);
  all_tensors["layer_30_gate_up_proj"] = layer_30_gate_up_proj;
  char *layer_30_gate_up_bias = (char*)(mpk_provider_ptrs[602]);
  all_tensors["layer_30_gate_up_bias"] = layer_30_gate_up_bias;
  char *layer_30_down_proj = (char*)(mpk_provider_ptrs[603]);
  all_tensors["layer_30_down_proj"] = layer_30_down_proj;
  char *layer_30_down_bias = (char*)(mpk_provider_ptrs[604]);
  all_tensors["layer_30_down_bias"] = layer_30_down_bias;
  char *layer_31_input_layernorm = (char*)(mpk_provider_ptrs[605]);
  all_tensors["layer_31_input_layernorm"] = layer_31_input_layernorm;
  char *layer_31_qkv_mxfp4 = (char*)(mpk_provider_ptrs[606]);
  all_tensors["layer_31_qkv_mxfp4"] = layer_31_qkv_mxfp4;
  char *layer_31_qkv_bias = (char*)(mpk_provider_ptrs[607]);
  all_tensors["layer_31_qkv_bias"] = layer_31_qkv_bias;
  char *layer_31_k_cache = (char*)(mpk_provider_ptrs[608]);
  all_tensors["layer_31_k_cache"] = layer_31_k_cache;
  char *layer_31_v_cache = (char*)(mpk_provider_ptrs[609]);
  all_tensors["layer_31_v_cache"] = layer_31_v_cache;
  char *layer_31_sinks = (char*)(mpk_provider_ptrs[610]);
  all_tensors["layer_31_sinks"] = layer_31_sinks;
  char *layer_31_o_bias = (char*)(mpk_provider_ptrs[611]);
  all_tensors["layer_31_o_bias"] = layer_31_o_bias;
  char *layer_31_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[612]);
  all_tensors["layer_31_o_proj_mxfp4"] = layer_31_o_proj_mxfp4;
  char *layer_31_post_attn_layernorm = (char*)(mpk_provider_ptrs[613]);
  all_tensors["layer_31_post_attn_layernorm"] = layer_31_post_attn_layernorm;
  char *layer_31_moe_gate = (char*)(mpk_provider_ptrs[614]);
  all_tensors["layer_31_moe_gate"] = layer_31_moe_gate;
  char *layer_31_router_bias = (char*)(mpk_provider_ptrs[615]);
  all_tensors["layer_31_router_bias"] = layer_31_router_bias;
  char *layer_31_gate_up_proj = (char*)(mpk_provider_ptrs[616]);
  all_tensors["layer_31_gate_up_proj"] = layer_31_gate_up_proj;
  char *layer_31_gate_up_bias = (char*)(mpk_provider_ptrs[617]);
  all_tensors["layer_31_gate_up_bias"] = layer_31_gate_up_bias;
  char *layer_31_down_proj = (char*)(mpk_provider_ptrs[618]);
  all_tensors["layer_31_down_proj"] = layer_31_down_proj;
  char *layer_31_down_bias = (char*)(mpk_provider_ptrs[619]);
  all_tensors["layer_31_down_bias"] = layer_31_down_bias;
  char *layer_32_input_layernorm = (char*)(mpk_provider_ptrs[620]);
  all_tensors["layer_32_input_layernorm"] = layer_32_input_layernorm;
  char *layer_32_qkv_mxfp4 = (char*)(mpk_provider_ptrs[621]);
  all_tensors["layer_32_qkv_mxfp4"] = layer_32_qkv_mxfp4;
  char *layer_32_qkv_bias = (char*)(mpk_provider_ptrs[622]);
  all_tensors["layer_32_qkv_bias"] = layer_32_qkv_bias;
  char *layer_32_k_cache = (char*)(mpk_provider_ptrs[623]);
  all_tensors["layer_32_k_cache"] = layer_32_k_cache;
  char *layer_32_v_cache = (char*)(mpk_provider_ptrs[624]);
  all_tensors["layer_32_v_cache"] = layer_32_v_cache;
  char *layer_32_sinks = (char*)(mpk_provider_ptrs[625]);
  all_tensors["layer_32_sinks"] = layer_32_sinks;
  char *layer_32_o_bias = (char*)(mpk_provider_ptrs[626]);
  all_tensors["layer_32_o_bias"] = layer_32_o_bias;
  char *layer_32_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[627]);
  all_tensors["layer_32_o_proj_mxfp4"] = layer_32_o_proj_mxfp4;
  char *layer_32_post_attn_layernorm = (char*)(mpk_provider_ptrs[628]);
  all_tensors["layer_32_post_attn_layernorm"] = layer_32_post_attn_layernorm;
  char *layer_32_moe_gate = (char*)(mpk_provider_ptrs[629]);
  all_tensors["layer_32_moe_gate"] = layer_32_moe_gate;
  char *layer_32_router_bias = (char*)(mpk_provider_ptrs[630]);
  all_tensors["layer_32_router_bias"] = layer_32_router_bias;
  char *layer_32_gate_up_proj = (char*)(mpk_provider_ptrs[631]);
  all_tensors["layer_32_gate_up_proj"] = layer_32_gate_up_proj;
  char *layer_32_gate_up_bias = (char*)(mpk_provider_ptrs[632]);
  all_tensors["layer_32_gate_up_bias"] = layer_32_gate_up_bias;
  char *layer_32_down_proj = (char*)(mpk_provider_ptrs[633]);
  all_tensors["layer_32_down_proj"] = layer_32_down_proj;
  char *layer_32_down_bias = (char*)(mpk_provider_ptrs[634]);
  all_tensors["layer_32_down_bias"] = layer_32_down_bias;
  char *layer_33_input_layernorm = (char*)(mpk_provider_ptrs[635]);
  all_tensors["layer_33_input_layernorm"] = layer_33_input_layernorm;
  char *layer_33_qkv_mxfp4 = (char*)(mpk_provider_ptrs[636]);
  all_tensors["layer_33_qkv_mxfp4"] = layer_33_qkv_mxfp4;
  char *layer_33_qkv_bias = (char*)(mpk_provider_ptrs[637]);
  all_tensors["layer_33_qkv_bias"] = layer_33_qkv_bias;
  char *layer_33_k_cache = (char*)(mpk_provider_ptrs[638]);
  all_tensors["layer_33_k_cache"] = layer_33_k_cache;
  char *layer_33_v_cache = (char*)(mpk_provider_ptrs[639]);
  all_tensors["layer_33_v_cache"] = layer_33_v_cache;
  char *layer_33_sinks = (char*)(mpk_provider_ptrs[640]);
  all_tensors["layer_33_sinks"] = layer_33_sinks;
  char *layer_33_o_bias = (char*)(mpk_provider_ptrs[641]);
  all_tensors["layer_33_o_bias"] = layer_33_o_bias;
  char *layer_33_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[642]);
  all_tensors["layer_33_o_proj_mxfp4"] = layer_33_o_proj_mxfp4;
  char *layer_33_post_attn_layernorm = (char*)(mpk_provider_ptrs[643]);
  all_tensors["layer_33_post_attn_layernorm"] = layer_33_post_attn_layernorm;
  char *layer_33_moe_gate = (char*)(mpk_provider_ptrs[644]);
  all_tensors["layer_33_moe_gate"] = layer_33_moe_gate;
  char *layer_33_router_bias = (char*)(mpk_provider_ptrs[645]);
  all_tensors["layer_33_router_bias"] = layer_33_router_bias;
  char *layer_33_gate_up_proj = (char*)(mpk_provider_ptrs[646]);
  all_tensors["layer_33_gate_up_proj"] = layer_33_gate_up_proj;
  char *layer_33_gate_up_bias = (char*)(mpk_provider_ptrs[647]);
  all_tensors["layer_33_gate_up_bias"] = layer_33_gate_up_bias;
  char *layer_33_down_proj = (char*)(mpk_provider_ptrs[648]);
  all_tensors["layer_33_down_proj"] = layer_33_down_proj;
  char *layer_33_down_bias = (char*)(mpk_provider_ptrs[649]);
  all_tensors["layer_33_down_bias"] = layer_33_down_bias;
  char *layer_34_input_layernorm = (char*)(mpk_provider_ptrs[650]);
  all_tensors["layer_34_input_layernorm"] = layer_34_input_layernorm;
  char *layer_34_qkv_mxfp4 = (char*)(mpk_provider_ptrs[651]);
  all_tensors["layer_34_qkv_mxfp4"] = layer_34_qkv_mxfp4;
  char *layer_34_qkv_bias = (char*)(mpk_provider_ptrs[652]);
  all_tensors["layer_34_qkv_bias"] = layer_34_qkv_bias;
  char *layer_34_k_cache = (char*)(mpk_provider_ptrs[653]);
  all_tensors["layer_34_k_cache"] = layer_34_k_cache;
  char *layer_34_v_cache = (char*)(mpk_provider_ptrs[654]);
  all_tensors["layer_34_v_cache"] = layer_34_v_cache;
  char *layer_34_sinks = (char*)(mpk_provider_ptrs[655]);
  all_tensors["layer_34_sinks"] = layer_34_sinks;
  char *layer_34_o_bias = (char*)(mpk_provider_ptrs[656]);
  all_tensors["layer_34_o_bias"] = layer_34_o_bias;
  char *layer_34_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[657]);
  all_tensors["layer_34_o_proj_mxfp4"] = layer_34_o_proj_mxfp4;
  char *layer_34_post_attn_layernorm = (char*)(mpk_provider_ptrs[658]);
  all_tensors["layer_34_post_attn_layernorm"] = layer_34_post_attn_layernorm;
  char *layer_34_moe_gate = (char*)(mpk_provider_ptrs[659]);
  all_tensors["layer_34_moe_gate"] = layer_34_moe_gate;
  char *layer_34_router_bias = (char*)(mpk_provider_ptrs[660]);
  all_tensors["layer_34_router_bias"] = layer_34_router_bias;
  char *layer_34_gate_up_proj = (char*)(mpk_provider_ptrs[661]);
  all_tensors["layer_34_gate_up_proj"] = layer_34_gate_up_proj;
  char *layer_34_gate_up_bias = (char*)(mpk_provider_ptrs[662]);
  all_tensors["layer_34_gate_up_bias"] = layer_34_gate_up_bias;
  char *layer_34_down_proj = (char*)(mpk_provider_ptrs[663]);
  all_tensors["layer_34_down_proj"] = layer_34_down_proj;
  char *layer_34_down_bias = (char*)(mpk_provider_ptrs[664]);
  all_tensors["layer_34_down_bias"] = layer_34_down_bias;
  char *layer_35_input_layernorm = (char*)(mpk_provider_ptrs[665]);
  all_tensors["layer_35_input_layernorm"] = layer_35_input_layernorm;
  char *layer_35_qkv_mxfp4 = (char*)(mpk_provider_ptrs[666]);
  all_tensors["layer_35_qkv_mxfp4"] = layer_35_qkv_mxfp4;
  char *layer_35_qkv_bias = (char*)(mpk_provider_ptrs[667]);
  all_tensors["layer_35_qkv_bias"] = layer_35_qkv_bias;
  char *layer_35_k_cache = (char*)(mpk_provider_ptrs[668]);
  all_tensors["layer_35_k_cache"] = layer_35_k_cache;
  char *layer_35_v_cache = (char*)(mpk_provider_ptrs[669]);
  all_tensors["layer_35_v_cache"] = layer_35_v_cache;
  char *layer_35_sinks = (char*)(mpk_provider_ptrs[670]);
  all_tensors["layer_35_sinks"] = layer_35_sinks;
  char *layer_35_o_bias = (char*)(mpk_provider_ptrs[671]);
  all_tensors["layer_35_o_bias"] = layer_35_o_bias;
  char *layer_35_o_proj_mxfp4 = (char*)(mpk_provider_ptrs[672]);
  all_tensors["layer_35_o_proj_mxfp4"] = layer_35_o_proj_mxfp4;
  char *layer_35_post_attn_layernorm = (char*)(mpk_provider_ptrs[673]);
  all_tensors["layer_35_post_attn_layernorm"] = layer_35_post_attn_layernorm;
  char *layer_35_moe_gate = (char*)(mpk_provider_ptrs[674]);
  all_tensors["layer_35_moe_gate"] = layer_35_moe_gate;
  char *layer_35_router_bias = (char*)(mpk_provider_ptrs[675]);
  all_tensors["layer_35_router_bias"] = layer_35_router_bias;
  char *layer_35_gate_up_proj = (char*)(mpk_provider_ptrs[676]);
  all_tensors["layer_35_gate_up_proj"] = layer_35_gate_up_proj;
  char *layer_35_gate_up_bias = (char*)(mpk_provider_ptrs[677]);
  all_tensors["layer_35_gate_up_bias"] = layer_35_gate_up_bias;
  char *layer_35_down_proj = (char*)(mpk_provider_ptrs[678]);
  all_tensors["layer_35_down_proj"] = layer_35_down_proj;
  char *layer_35_down_bias = (char*)(mpk_provider_ptrs[679]);
  all_tensors["layer_35_down_bias"] = layer_35_down_bias;
  char *model_norm_weight = (char*)(mpk_provider_ptrs[680]);
  all_tensors["model_norm_weight"] = model_norm_weight;
  char *lm_head_mxfp4 = (char*)(mpk_provider_ptrs[681]);
  all_tensors["lm_head_mxfp4"] = lm_head_mxfp4;
  char *lm_head_bias = (char*)(mpk_provider_ptrs[682]);
  all_tensors["lm_head_bias"] = lm_head_bias;
  all_tensors["nullptr"] = nullptr;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
  CUDA_CHECK(cudaDeviceSynchronize());
}

__device__ __forceinline__
void _execute_task(TaskDesc const* task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_EMBEDDING && task_desc->variant_id == 0) {
      kernel::embedding_kernel<bfloat16, 1, 2944, 2944>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0]);

  }
  else if (task_desc->task_type == TASK_ARGMAX_REDUCE && task_desc->variant_id == 0) {
      kernel::argmax_reduce_kernel<bfloat16, 1, 0, 248>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->output_ptrs[0],
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  }
}

__device__ __forceinline__
void _execute_gang_task(TaskDesc const* task_desc,
                        RuntimeConfig const &runtime_config,
                        int tile_idx) {
  if (task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 && task_desc->variant_id == 0) {
      kernel::gang_full_layer_fused_kernel_mi300<1, 64, 2944, 2880, 64, 8, 4096, 466, 8, 4096, 64, 8, 128, 1, 16, 4096, 128, 4, 2944, 2944, 128, 64, true, MPK_MAX_NUM_BATCHED_REQUESTS>(
      task_desc->input_ptrs,
      task_desc->output_ptrs,
      runtime_config.rope_cos_ptr,
      runtime_config.rope_sin_ptr,
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      10,
      512,
      4096,
      0.180337f,
      10,
      23,
      2944,
      16,
      184,
      128,
      23,
      42,
      31,
      tile_idx,
      (int)task_desc->task_metadata._linear_reserved);

  }
  else if (task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 && task_desc->variant_id == 1) {
      kernel::gang_full_layer_fused_kernel_mi300<1, 64, 2944, 2880, 64, 8, 4096, 466, 8, 4096, 64, 8, 0, 1, 16, 4096, 128, 4, 2944, 2944, 128, 64, true, MPK_MAX_NUM_BATCHED_REQUESTS>(
      task_desc->input_ptrs,
      task_desc->output_ptrs,
      runtime_config.rope_cos_ptr,
      runtime_config.rope_sin_ptr,
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      10,
      512,
      4096,
      0.180337f,
      10,
      23,
      2944,
      16,
      184,
      128,
      23,
      42,
      31,
      tile_idx,
      (int)task_desc->task_metadata._linear_reserved);

  }
  else if (task_desc->task_type == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300 && task_desc->variant_id == 0) {
      kernel::gang_rmsnorm_linear_mxfp4_bias_argmax_kernel<1, 64, 2944, 2880>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->input_ptrs[3],
      task_desc->input_ptrs[4],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      task_desc->output_ptrs[2],
      runtime_config.step[0] + 1,
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      393,
      31,
      201216,
      tile_idx);

  }
}

__device__ __forceinline__
void _execute_gang_task_ml(TaskDesc const* task_desc,
                        RuntimeConfig const &runtime_config,
                        int tile_idx) {
  if (task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 && task_desc->variant_id == 0) {
      kernel::gang_full_layer_fused_kernel_mi300_inl<1, 64, 2944, 2880, 64, 8, 4096, 466, 8, 4096, 64, 8, 128, 1, 16, 4096, 128, 4, 2944, 2944, 128, 64, true, MPK_MAX_NUM_BATCHED_REQUESTS>(
      task_desc->input_ptrs,
      task_desc->output_ptrs,
      runtime_config.rope_cos_ptr,
      runtime_config.rope_sin_ptr,
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      10,
      512,
      4096,
      0.180337f,
      10,
      23,
      2944,
      16,
      184,
      128,
      23,
      42,
      31,
      tile_idx,
      (int)task_desc->task_metadata._linear_reserved);

  }
  else if (task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 && task_desc->variant_id == 1) {
      kernel::gang_full_layer_fused_kernel_mi300_inl<1, 64, 2944, 2880, 64, 8, 4096, 466, 8, 4096, 64, 8, 0, 1, 16, 4096, 128, 4, 2944, 2944, 128, 64, true, MPK_MAX_NUM_BATCHED_REQUESTS>(
      task_desc->input_ptrs,
      task_desc->output_ptrs,
      runtime_config.rope_cos_ptr,
      runtime_config.rope_sin_ptr,
      runtime_config.qo_indptr_buffer,
      runtime_config.paged_kv_indptr_buffer,
      runtime_config.paged_kv_indices_buffer,
      runtime_config.paged_kv_last_page_len_buffer,
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      10,
      512,
      4096,
      0.180337f,
      10,
      23,
      2944,
      16,
      184,
      128,
      23,
      42,
      31,
      tile_idx,
      (int)task_desc->task_metadata._linear_reserved);

  }
  else if (task_desc->task_type == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300 && task_desc->variant_id == 0) {
      kernel::gang_rmsnorm_linear_mxfp4_bias_argmax_kernel<1, 64, 2944, 2880>(
      task_desc->input_ptrs[0],
      task_desc->input_ptrs[1],
      task_desc->input_ptrs[2],
      task_desc->input_ptrs[3],
      task_desc->input_ptrs[4],
      task_desc->output_ptrs[0],
      task_desc->output_ptrs[1],
      task_desc->output_ptrs[2],
      runtime_config.step[0] + 1,
      runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],
      393,
      31,
      201216,
      tile_idx);

  }
}

#define FLEET_MPK_ABI_VERSION 1
#define FLEET_MPK_NUM_PTRS 683
#define FLEET_MPK_BUILD_ID "fleet-mpk-20261019T041150Z"
static const char *const fleet_mpk_ptr_names[FLEET_MPK_NUM_PTRS] = {
  "input_token",
  "cos_position_embedding",
  "sin_position_embedding",
  "embed_out",
  "rmsnorm_out",
  "attn_in",
  "ck_fmha_q_workspace",
  "ck_fmha_lse_acc",
  "attn_out",
  "ck_fmha_o_acc",
  "attn_proj_out",
  "all_reduce_buf",
  "attn_allreduce_out",
  "rmsnorm_out_moe",
  "moe_gate_out",
  "moe_routing_indices",
  "moe_mask",
  "moe_topk_weight",
  "router_topk_counter",
  "oproj_topk_counters",
  "qkv_attn_barrier",
  "moe_fused_barrier",
  "swiglu_out",
  "mlp_out",
  "moe_workspace_f32",
  "moe_ws_zero",
  "mlp_weighted_sum_out",
  "mlp_final",
  "argmax_part_value",
  "argmax_part_index",
  "output_token",
  "splitk_workspace",
  "splitk_done_counter",
  "embed_tokens",
  "moe_workspace_f32_r1",
  "moe_workspace_f32_r2",
  "moe_workspace_f32_r3",
  "moe_workspace_f32_r4",
  "moe_workspace_f32_r5",
  "moe_workspace_f32_r6",
  "moe_workspace_f32_r7",
  "moe_workspace_f32_r8",
  "moe_workspace_f32_r9",
  "moe_workspace_f32_r10",
  "moe_workspace_f32_r11",
  "moe_workspace_f32_r12",
  "moe_workspace_f32_r13",
  "moe_workspace_f32_r14",
  "moe_workspace_f32_r15",
  "moe_workspace_f32_r16",
  "moe_workspace_f32_r17",
  "moe_workspace_f32_r18",
  "moe_workspace_f32_r19",
  "moe_workspace_f32_r20",
  "moe_workspace_f32_r21",
  "moe_workspace_f32_r22",
  "moe_workspace_f32_r23",
  "moe_workspace_f32_r24",
  "moe_workspace_f32_r25",
  "moe_workspace_f32_r26",
  "moe_workspace_f32_r27",
  "moe_workspace_f32_r28",
  "moe_workspace_f32_r29",
  "moe_workspace_f32_r30",
  "moe_workspace_f32_r31",
  "moe_workspace_f32_r32",
  "moe_workspace_f32_r33",
  "moe_workspace_f32_r34",
  "moe_workspace_f32_r35",
  "moe_workspace_f32_r36",
  "attn_out_r1",
  "attn_out_r2",
  "attn_out_r3",
  "attn_out_r4",
  "attn_out_r5",
  "attn_out_r6",
  "attn_out_r7",
  "attn_out_r8",
  "attn_out_r9",
  "attn_out_r10",
  "attn_out_r11",
  "attn_out_r12",
  "attn_out_r13",
  "attn_out_r14",
  "attn_out_r15",
  "attn_out_r16",
  "attn_out_r17",
  "attn_out_r18",
  "attn_out_r19",
  "attn_out_r20",
  "attn_out_r21",
  "attn_out_r22",
  "attn_out_r23",
  "attn_out_r24",
  "attn_out_r25",
  "attn_out_r26",
  "attn_out_r27",
  "attn_out_r28",
  "attn_out_r29",
  "attn_out_r30",
  "attn_out_r31",
  "attn_out_r32",
  "attn_out_r33",
  "attn_out_r34",
  "attn_out_r35",
  "attn_proj_out_r1",
  "attn_proj_out_r2",
  "attn_proj_out_r3",
  "attn_proj_out_r4",
  "attn_proj_out_r5",
  "attn_proj_out_r6",
  "attn_proj_out_r7",
  "attn_proj_out_r8",
  "attn_proj_out_r9",
  "attn_proj_out_r10",
  "attn_proj_out_r11",
  "attn_proj_out_r12",
  "attn_proj_out_r13",
  "attn_proj_out_r14",
  "attn_proj_out_r15",
  "attn_proj_out_r16",
  "attn_proj_out_r17",
  "attn_proj_out_r18",
  "attn_proj_out_r19",
  "attn_proj_out_r20",
  "attn_proj_out_r21",
  "attn_proj_out_r22",
  "attn_proj_out_r23",
  "attn_proj_out_r24",
  "attn_proj_out_r25",
  "attn_proj_out_r26",
  "attn_proj_out_r27",
  "attn_proj_out_r28",
  "attn_proj_out_r29",
  "attn_proj_out_r30",
  "attn_proj_out_r31",
  "attn_proj_out_r32",
  "attn_proj_out_r33",
  "attn_proj_out_r34",
  "attn_proj_out_r35",
  "layer_0_input_layernorm",
  "layer_0_qkv_mxfp4",
  "layer_0_qkv_bias",
  "layer_0_k_cache",
  "layer_0_v_cache",
  "layer_0_sinks",
  "layer_0_o_bias",
  "layer_0_o_proj_mxfp4",
  "layer_0_post_attn_layernorm",
  "layer_0_moe_gate",
  "layer_0_router_bias",
  "layer_0_gate_up_proj",
  "layer_0_gate_up_bias",
  "layer_0_down_proj",
  "layer_0_down_bias",
  "layer_1_input_layernorm",
  "layer_1_qkv_mxfp4",
  "layer_1_qkv_bias",
  "layer_1_k_cache",
  "layer_1_v_cache",
  "layer_1_sinks",
  "layer_1_o_bias",
  "layer_1_o_proj_mxfp4",
  "layer_1_post_attn_layernorm",
  "layer_1_moe_gate",
  "layer_1_router_bias",
  "layer_1_gate_up_proj",
  "layer_1_gate_up_bias",
  "layer_1_down_proj",
  "layer_1_down_bias",
  "layer_2_input_layernorm",
  "layer_2_qkv_mxfp4",
  "layer_2_qkv_bias",
  "layer_2_k_cache",
  "layer_2_v_cache",
  "layer_2_sinks",
  "layer_2_o_bias",
  "layer_2_o_proj_mxfp4",
  "layer_2_post_attn_layernorm",
  "layer_2_moe_gate",
  "layer_2_router_bias",
  "layer_2_gate_up_proj",
  "layer_2_gate_up_bias",
  "layer_2_down_proj",
  "layer_2_down_bias",
  "layer_3_input_layernorm",
  "layer_3_qkv_mxfp4",
  "layer_3_qkv_bias",
  "layer_3_k_cache",
  "layer_3_v_cache",
  "layer_3_sinks",
  "layer_3_o_bias",
  "layer_3_o_proj_mxfp4",
  "layer_3_post_attn_layernorm",
  "layer_3_moe_gate",
  "layer_3_router_bias",
  "layer_3_gate_up_proj",
  "layer_3_gate_up_bias",
  "layer_3_down_proj",
  "layer_3_down_bias",
  "layer_4_input_layernorm",
  "layer_4_qkv_mxfp4",
  "layer_4_qkv_bias",
  "layer_4_k_cache",
  "layer_4_v_cache",
  "layer_4_sinks",
  "layer_4_o_bias",
  "layer_4_o_proj_mxfp4",
  "layer_4_post_attn_layernorm",
  "layer_4_moe_gate",
  "layer_4_router_bias",
  "layer_4_gate_up_proj",
  "layer_4_gate_up_bias",
  "layer_4_down_proj",
  "layer_4_down_bias",
  "layer_5_input_layernorm",
  "layer_5_qkv_mxfp4",
  "layer_5_qkv_bias",
  "layer_5_k_cache",
  "layer_5_v_cache",
  "layer_5_sinks",
  "layer_5_o_bias",
  "layer_5_o_proj_mxfp4",
  "layer_5_post_attn_layernorm",
  "layer_5_moe_gate",
  "layer_5_router_bias",
  "layer_5_gate_up_proj",
  "layer_5_gate_up_bias",
  "layer_5_down_proj",
  "layer_5_down_bias",
  "layer_6_input_layernorm",
  "layer_6_qkv_mxfp4",
  "layer_6_qkv_bias",
  "layer_6_k_cache",
  "layer_6_v_cache",
  "layer_6_sinks",
  "layer_6_o_bias",
  "layer_6_o_proj_mxfp4",
  "layer_6_post_attn_layernorm",
  "layer_6_moe_gate",
  "layer_6_router_bias",
  "layer_6_gate_up_proj",
  "layer_6_gate_up_bias",
  "layer_6_down_proj",
  "layer_6_down_bias",
  "layer_7_input_layernorm",
  "layer_7_qkv_mxfp4",
  "layer_7_qkv_bias",
  "layer_7_k_cache",
  "layer_7_v_cache",
  "layer_7_sinks",
  "layer_7_o_bias",
  "layer_7_o_proj_mxfp4",
  "layer_7_post_attn_layernorm",
  "layer_7_moe_gate",
  "layer_7_router_bias",
  "layer_7_gate_up_proj",
  "layer_7_gate_up_bias",
  "layer_7_down_proj",
  "layer_7_down_bias",
  "layer_8_input_layernorm",
  "layer_8_qkv_mxfp4",
  "layer_8_qkv_bias",
  "layer_8_k_cache",
  "layer_8_v_cache",
  "layer_8_sinks",
  "layer_8_o_bias",
  "layer_8_o_proj_mxfp4",
  "layer_8_post_attn_layernorm",
  "layer_8_moe_gate",
  "layer_8_router_bias",
  "layer_8_gate_up_proj",
  "layer_8_gate_up_bias",
  "layer_8_down_proj",
  "layer_8_down_bias",
  "layer_9_input_layernorm",
  "layer_9_qkv_mxfp4",
  "layer_9_qkv_bias",
  "layer_9_k_cache",
  "layer_9_v_cache",
  "layer_9_sinks",
  "layer_9_o_bias",
  "layer_9_o_proj_mxfp4",
  "layer_9_post_attn_layernorm",
  "layer_9_moe_gate",
  "layer_9_router_bias",
  "layer_9_gate_up_proj",
  "layer_9_gate_up_bias",
  "layer_9_down_proj",
  "layer_9_down_bias",
  "layer_10_input_layernorm",
  "layer_10_qkv_mxfp4",
  "layer_10_qkv_bias",
  "layer_10_k_cache",
  "layer_10_v_cache",
  "layer_10_sinks",
  "layer_10_o_bias",
  "layer_10_o_proj_mxfp4",
  "layer_10_post_attn_layernorm",
  "layer_10_moe_gate",
  "layer_10_router_bias",
  "layer_10_gate_up_proj",
  "layer_10_gate_up_bias",
  "layer_10_down_proj",
  "layer_10_down_bias",
  "layer_11_input_layernorm",
  "layer_11_qkv_mxfp4",
  "layer_11_qkv_bias",
  "layer_11_k_cache",
  "layer_11_v_cache",
  "layer_11_sinks",
  "layer_11_o_bias",
  "layer_11_o_proj_mxfp4",
  "layer_11_post_attn_layernorm",
  "layer_11_moe_gate",
  "layer_11_router_bias",
  "layer_11_gate_up_proj",
  "layer_11_gate_up_bias",
  "layer_11_down_proj",
  "layer_11_down_bias",
  "layer_12_input_layernorm",
  "layer_12_qkv_mxfp4",
  "layer_12_qkv_bias",
  "layer_12_k_cache",
  "layer_12_v_cache",
  "layer_12_sinks",
  "layer_12_o_bias",
  "layer_12_o_proj_mxfp4",
  "layer_12_post_attn_layernorm",
  "layer_12_moe_gate",
  "layer_12_router_bias",
  "layer_12_gate_up_proj",
  "layer_12_gate_up_bias",
  "layer_12_down_proj",
  "layer_12_down_bias",
  "layer_13_input_layernorm",
  "layer_13_qkv_mxfp4",
  "layer_13_qkv_bias",
  "layer_13_k_cache",
  "layer_13_v_cache",
  "layer_13_sinks",
  "layer_13_o_bias",
  "layer_13_o_proj_mxfp4",
  "layer_13_post_attn_layernorm",
  "layer_13_moe_gate",
  "layer_13_router_bias",
  "layer_13_gate_up_proj",
  "layer_13_gate_up_bias",
  "layer_13_down_proj",
  "layer_13_down_bias",
  "layer_14_input_layernorm",
  "layer_14_qkv_mxfp4",
  "layer_14_qkv_bias",
  "layer_14_k_cache",
  "layer_14_v_cache",
  "layer_14_sinks",
  "layer_14_o_bias",
  "layer_14_o_proj_mxfp4",
  "layer_14_post_attn_layernorm",
  "layer_14_moe_gate",
  "layer_14_router_bias",
  "layer_14_gate_up_proj",
  "layer_14_gate_up_bias",
  "layer_14_down_proj",
  "layer_14_down_bias",
  "layer_15_input_layernorm",
  "layer_15_qkv_mxfp4",
  "layer_15_qkv_bias",
  "layer_15_k_cache",
  "layer_15_v_cache",
  "layer_15_sinks",
  "layer_15_o_bias",
  "layer_15_o_proj_mxfp4",
  "layer_15_post_attn_layernorm",
  "layer_15_moe_gate",
  "layer_15_router_bias",
  "layer_15_gate_up_proj",
  "layer_15_gate_up_bias",
  "layer_15_down_proj",
  "layer_15_down_bias",
  "layer_16_input_layernorm",
  "layer_16_qkv_mxfp4",
  "layer_16_qkv_bias",
  "layer_16_k_cache",
  "layer_16_v_cache",
  "layer_16_sinks",
  "layer_16_o_bias",
  "layer_16_o_proj_mxfp4",
  "layer_16_post_attn_layernorm",
  "layer_16_moe_gate",
  "layer_16_router_bias",
  "layer_16_gate_up_proj",
  "layer_16_gate_up_bias",
  "layer_16_down_proj",
  "layer_16_down_bias",
  "layer_17_input_layernorm",
  "layer_17_qkv_mxfp4",
  "layer_17_qkv_bias",
  "layer_17_k_cache",
  "layer_17_v_cache",
  "layer_17_sinks",
  "layer_17_o_bias",
  "layer_17_o_proj_mxfp4",
  "layer_17_post_attn_layernorm",
  "layer_17_moe_gate",
  "layer_17_router_bias",
  "layer_17_gate_up_proj",
  "layer_17_gate_up_bias",
  "layer_17_down_proj",
  "layer_17_down_bias",
  "layer_18_input_layernorm",
  "layer_18_qkv_mxfp4",
  "layer_18_qkv_bias",
  "layer_18_k_cache",
  "layer_18_v_cache",
  "layer_18_sinks",
  "layer_18_o_bias",
  "layer_18_o_proj_mxfp4",
  "layer_18_post_attn_layernorm",
  "layer_18_moe_gate",
  "layer_18_router_bias",
  "layer_18_gate_up_proj",
  "layer_18_gate_up_bias",
  "layer_18_down_proj",
  "layer_18_down_bias",
  "layer_19_input_layernorm",
  "layer_19_qkv_mxfp4",
  "layer_19_qkv_bias",
  "layer_19_k_cache",
  "layer_19_v_cache",
  "layer_19_sinks",
  "layer_19_o_bias",
  "layer_19_o_proj_mxfp4",
  "layer_19_post_attn_layernorm",
  "layer_19_moe_gate",
  "layer_19_router_bias",
  "layer_19_gate_up_proj",
  "layer_19_gate_up_bias",
  "layer_19_down_proj",
  "layer_19_down_bias",
  "layer_20_input_layernorm",
  "layer_20_qkv_mxfp4",
  "layer_20_qkv_bias",
  "layer_20_k_cache",
  "layer_20_v_cache",
  "layer_20_sinks",
  "layer_20_o_bias",
  "layer_20_o_proj_mxfp4",
  "layer_20_post_attn_layernorm",
  "layer_20_moe_gate",
  "layer_20_router_bias",
  "layer_20_gate_up_proj",
  "layer_20_gate_up_bias",
  "layer_20_down_proj",
  "layer_20_down_bias",
  "layer_21_input_layernorm",
  "layer_21_qkv_mxfp4",
  "layer_21_qkv_bias",
  "layer_21_k_cache",
  "layer_21_v_cache",
  "layer_21_sinks",
  "layer_21_o_bias",
  "layer_21_o_proj_mxfp4",
  "layer_21_post_attn_layernorm",
  "layer_21_moe_gate",
  "layer_21_router_bias",
  "layer_21_gate_up_proj",
  "layer_21_gate_up_bias",
  "layer_21_down_proj",
  "layer_21_down_bias",
  "layer_22_input_layernorm",
  "layer_22_qkv_mxfp4",
  "layer_22_qkv_bias",
  "layer_22_k_cache",
  "layer_22_v_cache",
  "layer_22_sinks",
  "layer_22_o_bias",
  "layer_22_o_proj_mxfp4",
  "layer_22_post_attn_layernorm",
  "layer_22_moe_gate",
  "layer_22_router_bias",
  "layer_22_gate_up_proj",
  "layer_22_gate_up_bias",
  "layer_22_down_proj",
  "layer_22_down_bias",
  "layer_23_input_layernorm",
  "layer_23_qkv_mxfp4",
  "layer_23_qkv_bias",
  "layer_23_k_cache",
  "layer_23_v_cache",
  "layer_23_sinks",
  "layer_23_o_bias",
  "layer_23_o_proj_mxfp4",
  "layer_23_post_attn_layernorm",
  "layer_23_moe_gate",
  "layer_23_router_bias",
  "layer_23_gate_up_proj",
  "layer_23_gate_up_bias",
  "layer_23_down_proj",
  "layer_23_down_bias",
  "layer_24_input_layernorm",
  "layer_24_qkv_mxfp4",
  "layer_24_qkv_bias",
  "layer_24_k_cache",
  "layer_24_v_cache",
  "layer_24_sinks",
  "layer_24_o_bias",
  "layer_24_o_proj_mxfp4",
  "layer_24_post_attn_layernorm",
  "layer_24_moe_gate",
  "layer_24_router_bias",
  "layer_24_gate_up_proj",
  "layer_24_gate_up_bias",
  "layer_24_down_proj",
  "layer_24_down_bias",
  "layer_25_input_layernorm",
  "layer_25_qkv_mxfp4",
  "layer_25_qkv_bias",
  "layer_25_k_cache",
  "layer_25_v_cache",
  "layer_25_sinks",
  "layer_25_o_bias",
  "layer_25_o_proj_mxfp4",
  "layer_25_post_attn_layernorm",
  "layer_25_moe_gate",
  "layer_25_router_bias",
  "layer_25_gate_up_proj",
  "layer_25_gate_up_bias",
  "layer_25_down_proj",
  "layer_25_down_bias",
  "layer_26_input_layernorm",
  "layer_26_qkv_mxfp4",
  "layer_26_qkv_bias",
  "layer_26_k_cache",
  "layer_26_v_cache",
  "layer_26_sinks",
  "layer_26_o_bias",
  "layer_26_o_proj_mxfp4",
  "layer_26_post_attn_layernorm",
  "layer_26_moe_gate",
  "layer_26_router_bias",
  "layer_26_gate_up_proj",
  "layer_26_gate_up_bias",
  "layer_26_down_proj",
  "layer_26_down_bias",
  "layer_27_input_layernorm",
  "layer_27_qkv_mxfp4",
  "layer_27_qkv_bias",
  "layer_27_k_cache",
  "layer_27_v_cache",
  "layer_27_sinks",
  "layer_27_o_bias",
  "layer_27_o_proj_mxfp4",
  "layer_27_post_attn_layernorm",
  "layer_27_moe_gate",
  "layer_27_router_bias",
  "layer_27_gate_up_proj",
  "layer_27_gate_up_bias",
  "layer_27_down_proj",
  "layer_27_down_bias",
  "layer_28_input_layernorm",
  "layer_28_qkv_mxfp4",
  "layer_28_qkv_bias",
  "layer_28_k_cache",
  "layer_28_v_cache",
  "layer_28_sinks",
  "layer_28_o_bias",
  "layer_28_o_proj_mxfp4",
  "layer_28_post_attn_layernorm",
  "layer_28_moe_gate",
  "layer_28_router_bias",
  "layer_28_gate_up_proj",
  "layer_28_gate_up_bias",
  "layer_28_down_proj",
  "layer_28_down_bias",
  "layer_29_input_layernorm",
  "layer_29_qkv_mxfp4",
  "layer_29_qkv_bias",
  "layer_29_k_cache",
  "layer_29_v_cache",
  "layer_29_sinks",
  "layer_29_o_bias",
  "layer_29_o_proj_mxfp4",
  "layer_29_post_attn_layernorm",
  "layer_29_moe_gate",
  "layer_29_router_bias",
  "layer_29_gate_up_proj",
  "layer_29_gate_up_bias",
  "layer_29_down_proj",
  "layer_29_down_bias",
  "layer_30_input_layernorm",
  "layer_30_qkv_mxfp4",
  "layer_30_qkv_bias",
  "layer_30_k_cache",
  "layer_30_v_cache",
  "layer_30_sinks",
  "layer_30_o_bias",
  "layer_30_o_proj_mxfp4",
  "layer_30_post_attn_layernorm",
  "layer_30_moe_gate",
  "layer_30_router_bias",
  "layer_30_gate_up_proj",
  "layer_30_gate_up_bias",
  "layer_30_down_proj",
  "layer_30_down_bias",
  "layer_31_input_layernorm",
  "layer_31_qkv_mxfp4",
  "layer_31_qkv_bias",
  "layer_31_k_cache",
  "layer_31_v_cache",
  "layer_31_sinks",
  "layer_31_o_bias",
  "layer_31_o_proj_mxfp4",
  "layer_31_post_attn_layernorm",
  "layer_31_moe_gate",
  "layer_31_router_bias",
  "layer_31_gate_up_proj",
  "layer_31_gate_up_bias",
  "layer_31_down_proj",
  "layer_31_down_bias",
  "layer_32_input_layernorm",
  "layer_32_qkv_mxfp4",
  "layer_32_qkv_bias",
  "layer_32_k_cache",
  "layer_32_v_cache",
  "layer_32_sinks",
  "layer_32_o_bias",
  "layer_32_o_proj_mxfp4",
  "layer_32_post_attn_layernorm",
  "layer_32_moe_gate",
  "layer_32_router_bias",
  "layer_32_gate_up_proj",
  "layer_32_gate_up_bias",
  "layer_32_down_proj",
  "layer_32_down_bias",
  "layer_33_input_layernorm",
  "layer_33_qkv_mxfp4",
  "layer_33_qkv_bias",
  "layer_33_k_cache",
  "layer_33_v_cache",
  "layer_33_sinks",
  "layer_33_o_bias",
  "layer_33_o_proj_mxfp4",
  "layer_33_post_attn_layernorm",
  "layer_33_moe_gate",
  "layer_33_router_bias",
  "layer_33_gate_up_proj",
  "layer_33_gate_up_bias",
  "layer_33_down_proj",
  "layer_33_down_bias",
  "layer_34_input_layernorm",
  "layer_34_qkv_mxfp4",
  "layer_34_qkv_bias",
  "layer_34_k_cache",
  "layer_34_v_cache",
  "layer_34_sinks",
  "layer_34_o_bias",
  "layer_34_o_proj_mxfp4",
  "layer_34_post_attn_layernorm",
  "layer_34_moe_gate",
  "layer_34_router_bias",
  "layer_34_gate_up_proj",
  "layer_34_gate_up_bias",
  "layer_34_down_proj",
  "layer_34_down_bias",
  "layer_35_input_layernorm",
  "layer_35_qkv_mxfp4",
  "layer_35_qkv_bias",
  "layer_35_k_cache",
  "layer_35_v_cache",
  "layer_35_sinks",
  "layer_35_o_bias",
  "layer_35_o_proj_mxfp4",
  "layer_35_post_attn_layernorm",
  "layer_35_moe_gate",
  "layer_35_router_bias",
  "layer_35_gate_up_proj",
  "layer_35_gate_up_bias",
  "layer_35_down_proj",
  "layer_35_down_bias",
  "model_norm_weight",
  "lm_head_mxfp4",
  "lm_head_bias"
};
#include "task_graph_blob.inc"

// ------------------------------------------------------------------------------------------------------------
// libAiterMK fleet MPK provider ABI (in place of the Python module of a standalone build)
// ------------------------------------------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#define FLEET_MPK_API extern "C" __attribute__((visibility("default")))

static const char *mpk_provider_task_graph_json(void) {
  return reinterpret_cast<const char *>(fleet_mpk_task_graph);
}

FLEET_MPK_API int fleet_mpk_abi_version(void) { return FLEET_MPK_ABI_VERSION; }
FLEET_MPK_API const char *fleet_mpk_build_id(void) { return FLEET_MPK_BUILD_ID; }
FLEET_MPK_API int fleet_mpk_num_ptrs(void) { return FLEET_MPK_NUM_PTRS; }
FLEET_MPK_API const char *fleet_mpk_ptr_name(int i) {
  return (i >= 0 && i < FLEET_MPK_NUM_PTRS) ? fleet_mpk_ptr_names[i] : nullptr;
}

// ptrs: one device address per generated pointer (fleet_mpk_ptr_name order); read only during this call.
// meta: the 10 runtime metadata buffers (step, tokens, input_tokens, output_tokens, num_new_tokens,
// prompt_lengths, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len).
FLEET_MPK_API int fleet_mpk_init(void *const *ptrs, int num_ptrs, void *const *meta, int num_meta,
                                 void *profiler_buffer, int my_rank, int num_workers,
                                 int num_local_schedulers, int num_remote_schedulers, int max_seq_length,
                                 int total_num_requests, long long eos_token_id) {
  if (ptrs == nullptr || num_ptrs != FLEET_MPK_NUM_PTRS || meta == nullptr || num_meta != 10) {
    return -1;
  }
  mpk_provider_ptrs = ptrs;
  std::vector<void *> meta_tensors(meta, meta + num_meta);
  init_persistent_kernel(meta_tensors, profiler_buffer, my_rank, num_workers, num_local_schedulers,
                         num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id);
  mpk_provider_ptrs = nullptr;
  return 0;
}
FLEET_MPK_API void fleet_mpk_set_rope_tables(void *cos_ptr, void *sin_ptr) { set_rope_tables(cos_ptr, sin_ptr); }
FLEET_MPK_API void fleet_mpk_set_max_seq_length(int max_seq_length) { set_max_seq_length(max_seq_length); }
FLEET_MPK_API void fleet_mpk_init_request(void) { init_request_resources(); }
FLEET_MPK_API void fleet_mpk_reset_decode_progress(void) { reset_decode_progress(); }
// End the running launch at its next 16-iteration check (builds with MPK_HOST_STOP; a no-op otherwise).
FLEET_MPK_API void fleet_mpk_request_stop(void) { request_decode_stop(); }
// Blocks until the request's launch ends; run it on a thread of its own.
FLEET_MPK_API void fleet_mpk_launch(void *stream) { launch_persistent_kernel(reinterpret_cast<hipStream_t>(stream)); }
FLEET_MPK_API int fleet_mpk_decode_progress(int slot) { return get_decode_progress(slot); }
FLEET_MPK_API int fleet_mpk_decode_token(int pos) { return get_decode_token(pos); }
FLEET_MPK_API void fleet_mpk_finalize(void) { finalize_persistent_kernel(); }
