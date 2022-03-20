#ifndef SISB_H
#define SISB_H

#include <unordered_map>
#include "prefetcher.h"
using namespace std;

class SISBPrefetcher : public Prefetcher
{
private:
    /* architectural metadata */

    // training unit (maps pc to last address)
    unordered_map<uint64_t, uint64_t> tu;

    // mapping cache (maps address to next address)
    unordered_map<uint64_t, uint64_t> cache;

    /* performance metadata */
    unordered_map<uint64_t, uint64_t> outstanding;
    unordered_map<uint64_t, uint32_t> issued;
    unordered_map<uint64_t, uint32_t> untimely;
    unordered_map<uint64_t, uint32_t> accurate;
    unordered_map<uint64_t, uint32_t> total;
    uint64_t divergence = 0;
    
    void init_knobs();
	void init_stats();
    uint64_t get_prediction(uint64_t pc, uint64_t addr);
    
public:
    SISBPrefetcher(string type);
    ~SISBPrefetcher();
	void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t> &pref_addr);
    void register_fill(uint64_t address);
    void dump_stats();
	void print_config();
    //void l2c_notify_useful(uint64_t addr, uint64_t pc);
};


#endif /* SISB_H */