#ifndef TRIAGE_H
#define TRIAGE_H

#include <unordered_map>
#include "cache.h"
#include "champsim.h"
#include "prefetcher.h"
#include "triage_core.h"

using namespace std;

struct TriageConfig {
    int lookahead;
    int degree;

    int on_chip_set, on_chip_assoc;
    int training_unit_size;
    bool use_dynamic_assoc;

    TriageReplType repl;
};

class TriagePrefetcher : public Prefetcher
{
private:
    CACHE *m_parent_cache;
    TriageConfig conf[NUM_CPUS];
    TriageCore data[NUM_CPUS];
    uint64_t last_address[NUM_CPUS];
     
    void init_knobs();
	void init_stats();
    uint64_t get_prediction(uint64_t pc, uint64_t addr);
    
public:
    TriagePrefetcher(string type, CACHE *cache);
    ~TriagePrefetcher();
	void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t> &pref_addr, vector<uint64_t> &pref_level);
    void register_fill(uint64_t address, uint8_t prefetch, uint32_t metadata_in);
    void dump_stats();
	void print_config();
};


#endif /* TRIAGE_H */