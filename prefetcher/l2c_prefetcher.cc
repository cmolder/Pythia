#include "cache.h"

void CACHE::l2c_prefetcher_initialize() 
{

}

uint32_t CACHE::l2c_prefetcher_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in, uint64_t instr_id, uint64_t curr_cycle)
{
  return metadata_in;
}

uint32_t CACHE::l2c_prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::l2c_prefetcher_cycle_operate() 
{

}

void CACHE::l2c_prefetcher_final_stats()
{

}

uint32_t CACHE::l2c_prefetcher_prefetch_hit(uint64_t addr, uint64_t ip, uint32_t metadata_in)
{
    return metadata_in;
}

void CACHE::l2c_prefetcher_print_config()
{
	
}

void CACHE::l2c_prefetcher_broadcast_bw(uint8_t bw_level)
{

}

void CACHE::l2c_prefetcher_broadcast_ipc(uint8_t ipc)
{

}

void CACHE::l2c_prefetcher_broadcast_acc(uint32_t acc_level)
{

}
