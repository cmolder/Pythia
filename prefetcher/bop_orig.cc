/*
 * Best Offset prefetcher as implemented by Pierre Michaud.
 *
 * Reference: https://github.com/Quangmire/ChampSim/blob/master/prefetcher/bo.h
 */

#include "bop_orig.h"
#include "bop_orig_helper.h"
#include "cache.h"
#include "prefetcher.h"

namespace knob
{
	extern uint32_t bop_pref_degree;
}

void BOOriginalPrefetcher::init_knobs() {

}

void BOOriginalPrefetcher::init_stats() {

}

BOOriginalPrefetcher::BOOriginalPrefetcher(std::string type, CACHE *cache) : 
    Prefetcher(type),
    m_parent_cache(cache) {

    init_knobs();
    init_stats();

    bo_prefetcher_initialize();
}

BOOriginalPrefetcher::~BOOriginalPrefetcher() {

}

void BOOriginalPrefetcher::register_fill(uint64_t address, uint32_t set, uint32_t way, 
                                         uint8_t prefetch, uint64_t evicted_addr) {
    bo_prefetcher_cache_fill(address, set, way, prefetch, evicted_addr);
}

void BOOriginalPrefetcher::invoke_prefetcher(
    uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, 
    std::vector<uint64_t> &pref_addr, std::vector<uint64_t> &pref_level) {


    // if(instr_id == 0)
    //     return;

    std::vector<uint64_t> bo_candidates;
    bo_prefetcher_operate(
        address, pc, cache_hit, type, 
        m_parent_cache->get_set(address), 
        m_parent_cache->get_way(address, m_parent_cache->get_set(address)), 
        bo_candidates);

    for (uint32_t i = 0; i < bo_candidates.size(); i++) {
        pref_addr.push_back(bo_candidates[i]);
        pref_level.push_back(0);
    }
}

void BOOriginalPrefetcher::dump_stats() {
    bo_prefetcher_final_stats();
}

void BOOriginalPrefetcher::print_config() {
    // Print BOP configurable knobs, and those represented by 
    // a hardcoded value.
    cout << "bop_orig stats:" << endl
         << "bop_max_rounds " << knob::bop_max_rounds << endl
         << "bop_max_score " << knob::bop_max_score << endl
         //<< "bop_top_n " << endl
         //<< "bop_enable_pref_buffer " << endl
         //<< "bop_pref_buffer_size " << endl
         << "bop_pref_degree " << knob::bop_pref_degree << endl
         //<< "bop_rr_size " << endl
         << "bop_candidates (OFFSET) ";
    for(uint32_t index = 0; index < NOFFSETS; ++index) {
		cout << OFFSET[index] << ",";
	}
    // Print original BOP hard-coded knobs.
    // TODO: Remove hard-coded knobs
    cout << endl
         << "bop_orig DEFAULT_OFFSET " << DEFAULT_OFFSET << endl
         << "bop_orig RRINDEX " << RRINDEX << endl
         << "bop_orig RRTAG " << RRTAG << endl
         << "bop_orig DELAYQSIZE " << DELAYQSIZE << endl
         << "bop_orig DELAY " << DELAY << endl
         << "bop_orig TIME_BITS " << TIME_BITS << endl
         << "bop_orig LOW_SCORE " << LOW_SCORE << endl
         << "bop_orig BAD_SCORE " << BAD_SCORE << endl
         << endl;
}