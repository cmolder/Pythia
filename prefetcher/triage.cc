#include <iostream>
#include "triage.h"
#include "triage_onchip.h"
#include "triage_core.h"

#include "cache.h"
#include "champsim.h"
#include "memory_class.h"

namespace knob
{
    extern uint32_t       triage_lookahead;
    extern uint32_t       triage_degree;
    extern uint32_t       triage_on_chip_set;       // 262144 (1MB)
    extern uint32_t       triage_on_chip_assoc;      // 8
    extern uint32_t       triage_training_unit_size; // 10000000
    extern TriageReplType triage_repl;               // TABLEISB_REPL_HAWKEYE, or TABLEISB_REPL_LRU or TABLEISB_REPL_PERFECT
    extern bool           triage_use_dynamic_assoc;  // true
    extern uint32_t       triage_max_allowed_degree; // 8
}

void TriagePrefetcher::init_knobs() 
{
    for (uint32_t cpu = 0; cpu < NUM_CPUS; cpu++) {
        conf[cpu].lookahead = knob::triage_lookahead;
        conf[cpu].degree = knob::triage_degree;
        conf[cpu].on_chip_set = knob::triage_on_chip_set; 
        conf[cpu].on_chip_assoc = knob::triage_on_chip_assoc;
        conf[cpu].training_unit_size = knob::triage_training_unit_size;
        conf[cpu].repl = knob::triage_repl;
        conf[cpu].use_dynamic_assoc = knob::triage_use_dynamic_assoc;
        
        data[cpu].set_conf(&conf[cpu]);
    }
}

void TriagePrefetcher::init_stats() 
{
    //bzero(&stats, sizeof(stats));
}

TriagePrefetcher::TriagePrefetcher(string type, CACHE *cache) : Prefetcher(type)
{
    m_parent_cache = cache;
    init_knobs();
    init_stats();
}

TriagePrefetcher::~TriagePrefetcher() 
{

}

void TriagePrefetcher::invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t> &pref_addr, vector<uint64_t> &pref_level)
{
    if (type != LOAD)
        return;
    
    address = (address >> 6) << 6;
    if(address == last_address[m_parent_cache->cpu])
        return;
    last_address[m_parent_cache->cpu] = address;
    
    uint32_t i;
    uint64_t prefetch_addr_list[knob::triage_max_allowed_degree];
    for (i = 0; i < knob::triage_max_allowed_degree; ++i) {
        prefetch_addr_list[i] = 0;
    }
    data[m_parent_cache->cpu].calculatePrefetch(pc, address, cache_hit, prefetch_addr_list, 
                                                knob::triage_max_allowed_degree, m_parent_cache->cpu);

    int prefetched = 0;
    for (i = 0; i < knob::triage_max_allowed_degree; ++i) {
        if (prefetch_addr_list[i] == 0) {
            break;
        }
        //pref_addr.push_back(prefetch_addr_list[i]);
        //pref_level.push_back(fill_level)
        int ret = m_parent_cache->prefetch_line(pc, address, prefetch_addr_list[i], m_parent_cache->fill_level, address);
        
        // Add prefetches to pref_addr and pref_level vectors.
        // multi.*_pref will not double-prefetch in this case, as it will
        // recognize the requests are coming from Triage.
        pref_addr.push_back(prefetch_addr_list[i]);
        pref_level.push_back(m_parent_cache->fill_level);
        
        // Stop early if we surpass confidence ?
        if(ret)
        {
            prefetched++;
            if(prefetched >= conf[m_parent_cache->cpu].degree)
                break;
        }    
    }
    // Set cache assoc if dynamic
//    if (conf[cpu].use_dynamic_assoc) {
//        cout << "LLC WAY: " << LLC_WAY << ", ASSOC: " << data[cpu].get_assoc() << endl;
//        current_assoc = LLC_WAY - data[cpu].get_assoc();
//    }
    unsigned total_assoc = 0;
    for (uint32_t mycpu = 0; mycpu < NUM_CPUS; ++mycpu) {
        total_assoc += data[mycpu].get_assoc();
    }
    total_assoc /= NUM_CPUS;
    assert (total_assoc < LLC_WAY);
    //if (conf[cpu].repl != TRIAGE_REPL_PERFECT)
    //    current_assoc = LLC_WAY - total_assoc;

    return;
    
}

void TriagePrefetcher::register_fill(uint64_t address, uint8_t prefetch, uint32_t metadata_in)
{
    if(prefetch) {
       uint64_t next_addr;
       bool next_addr_exists = data[m_parent_cache->cpu].on_chip_data.get_next_addr(metadata_in, next_addr, 0, true);
       assert(next_addr_exists);
       //cout << "Filled " << hex << addr << "  by " << metadata_in << endl;
    }
}

void TriagePrefetcher::dump_stats() {
    // TODO - If it prints multiple times, use a flag variable to disbale printing on 2nd and later tries.
    for(uint32_t cpu=0; cpu < NUM_CPUS; cpu++) {
        cout << "Triage stats (CPU "  << cpu << ")" << endl;
        data[cpu].print_stats();
    }
}

void TriagePrefetcher::print_config()
{
    for(uint32_t cpu=0; cpu < NUM_CPUS; cpu++) {
        cout << "Triage configuration (CPU "  << cpu << ")" << endl;
        data[cpu].print_conf();
    }
}
