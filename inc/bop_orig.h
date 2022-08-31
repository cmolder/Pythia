/*
 * Best Offset prefetcher as implemented by Pierre Michaud.
 *
 * Reference: https://github.com/Quangmire/ChampSim/blob/master/prefetcher/bo.h
 */

#ifndef BOP_ORIG_H
#define BOP_ORIG_H

#include <stdio.h>
#include <stdlib.h>
#include "cache.h"
#include <map>


// Boilerplate for interacting with Pythia prefetcher class.
class BOOriginalPrefetcher : public Prefetcher {
private:

private:
	void init_knobs();
	void init_stats();
    CACHE* m_parent_cache;

public:
	BOOriginalPrefetcher(std::string type, CACHE* cache);
	~BOOriginalPrefetcher();
	void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, 
                           uint8_t type, std::vector<uint64_t> &pref_addr, 
                           std::vector<uint64_t> &pref_level);
	void register_fill(uint64_t address, uint32_t set, uint32_t way, 
                       uint8_t prefetch, uint64_t evicted_addr);
	void dump_stats();
	void print_config();
};

#endif // BOP_ORIG_H