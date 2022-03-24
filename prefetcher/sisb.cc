#include <iostream>
#include "sisb.h"
#include "champsim.h"
#include "memory_class.h"

namespace knob
{
    extern uint32_t sisb_pref_degree;
}

void SISBPrefetcher::init_knobs() 
{

}

void SISBPrefetcher::init_stats() 
{
    //bzero(&stats, sizeof(stats));
}

SISBPrefetcher::SISBPrefetcher(string type) : Prefetcher(type)
{
    init_knobs();
    init_stats();
}

SISBPrefetcher::~SISBPrefetcher() 
{

}

void SISBPrefetcher::invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t> &pref_addr)
{
    assert(pref_addr.size() == 0);
    // only consider demand misses
    if (type != LOAD)
        return;

    // initialize values
    if (issued.find(pc) == issued.end()) {
        issued[pc] = 0;
        untimely[pc] = 0;
        accurate[pc] = 0;
        total[pc] = 0;
    }
    total[pc]++;

    uint64_t addr_B = address >> LOG2_BLOCK_SIZE;

    /* training */
    if (tu.find(pc) != tu.end()) {
        uint64_t last_addr = tu[pc];
        if (cache.find(last_addr) != cache.end() && cache[last_addr] != addr_B) {
            divergence++;
        }
        cache[last_addr] = addr_B;
    }
    tu[pc] = addr_B;

    /* prediction */
    uint64_t pred = get_prediction(pc, addr_B);
    uint32_t count = 0;
    while (count < knob::sisb_pref_degree && pred != 0) {
        // issue prefetch
        pred <<= LOG2_BLOCK_SIZE;
        //int was_issued = parent->prefetch_line(pc, addr, pred, FILL_LLC, 0);
        pref_addr.push_back(pred);
        pred >>= LOG2_BLOCK_SIZE;
        //if (was_issued) {
            outstanding[pred] = pc;
            issued[pc]++;
            count++;
        //}
        
        // get next prediction
        pred = get_prediction(pc, pred);
    }
}

uint64_t SISBPrefetcher::get_prediction(uint64_t pc, uint64_t addr) 
{
    /* get most specific prediction possible */
    if (cache.find(addr) != cache.end()) {
        return cache[addr];
    } else {
        return 0;
    }
}

// void SISBPrefetcher::l2c_notify_useful(uint64_t addr, uint64_t pc) 
// {
//     accurate[pc]++;
// }

void SISBPrefetcher::register_fill(uint64_t address)
{
    uint64_t addr_B = address >> LOG2_BLOCK_SIZE;
    if (outstanding.find(addr_B) != outstanding.end()) {
        //uint64_t last_pc = outstanding[addr_B];
        //if (prefetch) {
        //} else {
        //    untimely[last_pc]++;    
        //}
        outstanding.erase(addr_B);
    }
}

void SISBPrefetcher::dump_stats() {
    // TODO - If it prints multiple times, use a flag variable to disbale printing on 2nd and later tries.
    cout << "performance stats:" << endl;
    uint32_t mappings = cache.size();
    cout << "mapping cache size: " << (mappings) << endl;
    cout << "divergence: " << divergence << endl;
}


void SISBPrefetcher::print_config()
{
    cout << "sisb_pref_degree " << knob::sisb_pref_degree << endl;
}
