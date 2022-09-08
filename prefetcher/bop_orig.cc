/*
 * Best Offset prefetcher as implemented by Pierre Michaud.
 *
 * Reference: https://github.com/Quangmire/ChampSim/blob/master/prefetcher/bo.h
 */

#include "bop_orig.h"

#include "bop_orig_helper.h"
#include "cache.h"
#include "prefetcher.h"

namespace knob {
extern vector<int32_t> bop_candidates;
extern int32_t bop_default_candidate;
extern uint32_t bop_max_rounds;
extern uint32_t bop_max_score;
extern uint32_t bop_low_score;
extern uint32_t bop_bad_score;
extern uint32_t bop_pref_degree;
}  // namespace knob

void BOOriginalPrefetcher::init_knobs() {}

void BOOriginalPrefetcher::init_stats() {}

BOOriginalPrefetcher::BOOriginalPrefetcher(std::string type, CACHE *cache)
    : Prefetcher(type), m_parent_cache(cache) {
  init_knobs();
  init_stats();

  bo_prefetcher_initialize(m_parent_cache->cache_type);
}

BOOriginalPrefetcher::~BOOriginalPrefetcher() {}

void BOOriginalPrefetcher::register_fill(uint64_t address, uint32_t set,
                                         uint32_t way, uint8_t prefetch,
                                         uint64_t evicted_addr) {
  bo_prefetcher_cache_fill(address, set, way, prefetch, evicted_addr);
}

void BOOriginalPrefetcher::invoke_prefetcher(
    uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type,
    std::vector<uint64_t> &pref_addr, std::vector<uint64_t> &pref_level) {
  // if(instr_id == 0)
  //     return;

  std::vector<uint64_t> bo_candidates;
  bo_prefetcher_operate(
      address, pc, cache_hit, type, m_parent_cache->get_set(address),
      m_parent_cache->get_way(address, m_parent_cache->get_set(address)),
      bo_candidates);

  for (uint32_t i = 0; i < bo_candidates.size(); i++) {
    pref_addr.push_back(bo_candidates[i]);
    pref_level.push_back(0);
  }
}

void BOOriginalPrefetcher::dump_stats() { bo_prefetcher_final_stats(); }

void BOOriginalPrefetcher::print_config() {
  // Print BOP configurable knobs, and those represented by
  // a hardcoded value.
  std::cout << "bop_candidates ";
  for (uint32_t index = 0; index < knob::bop_candidates.size(); index++) {
    std::cout << knob::bop_candidates[index] << ",";
  }
  std::cout << std::endl
       << "bop_default_candidate " << knob::bop_default_candidate << std::endl
       << "bop_max_rounds " << knob::bop_max_rounds << std::endl
       << "bop_max_score " << knob::bop_max_score << std::endl
       << "bop_low_score " << knob::bop_low_score << std::endl
       << "bop_bad_score " << knob::bop_bad_score << std::endl
       << "bop_pref_degree " << knob::bop_pref_degree << std::endl;

  // Print original BOP hard-coded knobs.
  // TODO: Remove hard-coded knobs
  std::cout << "bop RRINDEX " << RRINDEX << std::endl
            << "bop RRTAG " << RRTAG << std::endl
            << "bop DELAYQSIZE " << DELAYQSIZE << std::endl
            << "bop DELAY " << DELAY << std::endl
            << "bop TIME_BITS " << TIME_BITS << std::endl
            << std::endl;
}