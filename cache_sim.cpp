#include<iostream>
#include <cmath>
#include<cstdint>
#include<vector>
#include<algorithm>
#include<string>
#include <random>
#include <limits>

const uint64_t BLOCK_SIZE = 64;
const int CACHE_LINE = 4;

struct CacheLine {
	bool valid = 0;
	uint64_t tag = 0;
	uint64_t block;
	int insertedAt = -1;
	int lastUsedAt;
};

struct Result {
	uint64_t misses = 0;
	uint64_t hits = 0;
	uint64_t evictions = 0;
};

const int NUM_BLOCKS = 7;
const int NUM_STATES = 128 * NUM_BLOCKS;
const int NUM_ACTIONS = NUM_BLOCKS;

const int EPISODES = 5000;

const double ALPHA = 0.1;
const double GAMMA = 0.9;
const double EPSILON_START = 0.3;
const double EPSILON_MIN = 0.05;

std::vector<std::vector<double>> qtable(
    NUM_STATES,
    std::vector<double>(NUM_ACTIONS, 0.0)
);

int get_cache_mask(const std::vector<CacheLine>& cache) {
    int mask = 0;

    for (const CacheLine& c : cache) {
        if (c.valid && c.block < NUM_BLOCKS) {
            mask |= (1 << static_cast<int>(c.block));
        }
    }

    return mask;
}

int get_state(const std::vector<CacheLine>& cache, int requested_block) {
    int cache_mask = get_cache_mask(cache);

    return cache_mask * NUM_BLOCKS + requested_block;
}

void update_age(std::vector<int>& age, int hit_line) {
    int old_age = age[hit_line];

    for (int i = 0; i < 4; i++) {
        if (age[i] > old_age) {
            age[i]--;
        }
    }

    age[hit_line] = 3;
}

int choose_action(const std::vector<std::vector<double>>& q, int state, double epsilon, std::mt19937& rng) {
    if (state < 0 || state >= static_cast<int>(q.size())) {
        return -1;
    }
    int cache_mask = state / NUM_BLOCKS;
    std::vector<int> legal_actions;

    for (int block = 0; block < NUM_BLOCKS; block++) {
        if (cache_mask & (1 << block)) {
            legal_actions.push_back(block);
        }
    }

    if (legal_actions.empty()) {
        return -1;
    }

    std::uniform_real_distribution<double> probability(0.0, 1.0);

    if (probability(rng) < epsilon) {
        std::uniform_int_distribution<int> pick(
            0, static_cast<int>(legal_actions.size()) - 1
        );

        return legal_actions[pick(rng)];
    }

    double best_q = -std::numeric_limits<double>::infinity();
    std::vector<int> best_actions;

    for (int action : legal_actions) {
        double value = q[state][action];

        if (value > best_q + 1e-12) {
            best_q = value;
            best_actions.clear();
            best_actions.push_back(action);
        } else if (std::abs(value - best_q) < 1e-12) {
            best_actions.push_back(action);
        }
    }

    std::uniform_int_distribution<int> pick(0, static_cast<int>(best_actions.size()) - 1
    );

    return best_actions[pick(rng)];
}


void update_q(
    std::vector<std::vector<double>>& q,
    int state,
    int action,
    double reward,
    int next_state,
    double alpha,
    double discount_factor,
    bool terminal
) {
    if (state < 0 || state >= static_cast<int>(q.size()) ||
        action < 0 || action >= NUM_ACTIONS) {
        std::cerr << "Invalid Q-learning state/action\\n";
        return;
    }

    double old_q = q[state][action];
    double max_next_q = 0.0;

    if (!terminal && next_state >= 0 && next_state < static_cast<int>(q.size())) {
        int next_mask = next_state / NUM_BLOCKS;
        bool found = false;

        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (next_mask & (1 << a)) {
                if (!found || q[next_state][a] > max_next_q) {
                    max_next_q = q[next_state][a];
                    found = true;
                }
            }
        }
    }

    double target = reward;

    if (!terminal && next_state >= 0) {
        target += discount_factor * max_next_q;
    }

    q[state][action] = old_q + alpha * (target - old_q);
}


void train_q_learning(const std::vector<uint64_t>& training_addresses) {
    std::mt19937 rng(42);

    for (int episode = 0; episode < EPISODES; episode++) {
        std::vector<CacheLine> cache(CACHE_LINE);

        bool pending = false;
        int pending_state = -1;
        int pending_action = -1;
        double pending_reward = 0.0;
        double pending_discount = 1.0;

        int time = 0;

        double progress = static_cast<double>(episode) / EPISODES;

		double epsilon = std::max(EPSILON_MIN, EPSILON_START - (EPSILON_START - EPSILON_MIN) * progress);

        for (uint64_t address : training_addresses) {
            int block = static_cast<int>(address / BLOCK_SIZE);

            bool hit = false;

            for (CacheLine& c : cache) {
                if (c.valid && c.block == static_cast<uint64_t>(block)) {
                    hit = true;
                    c.lastUsedAt = time;
                    break;
                }
            }

            if (hit) {
                if (pending) {
                    pending_reward += pending_discount;
                    pending_discount *= GAMMA;
                }

                time++;
                continue;
            }

            if (pending) {
                pending_reward -= pending_discount;
                pending_discount *= GAMMA;
            }

            int empty_line = -1;

            for (int i = 0; i < CACHE_LINE; i++) {
                if (!cache[i].valid) {
                    empty_line = i;
                    break;
                }
            }

            if (empty_line != -1) {
                cache[empty_line].valid = true;
                cache[empty_line].block = block;
                cache[empty_line].insertedAt = time;
                cache[empty_line].lastUsedAt = time;

                time++;
                continue;
            }

            int state = get_state(cache, block);

            if (pending) {
                update_q(
                    qtable,
                    pending_state,
                    pending_action,
                    pending_reward,
                    state,
                    ALPHA,
                    pending_discount,
                    false
                );

                pending = false;
                pending_reward = 0.0;
                pending_discount = 1.0;
            }

            // Choose which resident block to evict.
            int victim_block = choose_action(
                qtable, state, epsilon, rng
            );

            // Find the cache line containing that block.
            int victim_line = -1;

            for (int i = 0; i < CACHE_LINE; i++) {
                if (cache[i].valid &&
                    cache[i].block ==
                        static_cast<uint64_t>(victim_block)) {
                    victim_line = i;
                    break;
                }
            }

            if (victim_line == -1) {
                std::cerr << "Invalid victim action\n";
                return;
            }

            // Replace the victim with the requested block.
            cache[victim_line].block = block;
            cache[victim_line].insertedAt = time;
            cache[victim_line].lastUsedAt = time;

            // Start collecting feedback for this action.
            pending = true;
            pending_state = state;
            pending_action = victim_block;
            pending_reward = 0.0;
            pending_discount = 1.0;

            time++;
        }

        if (pending) {
            update_q(
                qtable,
                pending_state,
                pending_action,
                pending_reward,
                -1,
                ALPHA,
                pending_discount,
                true
            );
        }
    }
}

Result simulate_q_learning( const std::vector<uint64_t>& addresses) {
    std::vector<CacheLine> cache(CACHE_LINE);
    Result result;

    std::mt19937 rng(123);

    for (uint64_t address : addresses) {
        int block = static_cast<int>(address / BLOCK_SIZE);

        bool hit = false;

        for (CacheLine& c : cache) {
            if (c.valid && c.block == static_cast<uint64_t>(block)) {
                hit = true;
                result.hits++;
                break;
            }
        }

        if (hit) {
            continue;
        }

        result.misses++;

        int empty_line = -1;

        for (int i = 0; i < CACHE_LINE; i++) {
            if (!cache[i].valid) {
                empty_line = i;
                break;
            }
        }

        if (empty_line != -1) {
            cache[empty_line].valid = true;
            cache[empty_line].block = block;
            continue;
        }

        int state = get_state(cache, block);

        int victim_block = choose_action(
            qtable, state, 0.0, rng
        );

        if (victim_block < 0) {
            std::cerr << "No legal Q-learning victim during evaluation\\n";
            return result;
        }

        int victim_line = -1;

        for (int i = 0; i < CACHE_LINE; i++) {
            if (cache[i].valid &&
                cache[i].block ==
                    static_cast<uint64_t>(victim_block)) {
                victim_line = i;
                break;
            }
        }


        if (victim_line < 0) {
            std::cerr << "Invalid Q-learning victim during evaluation\\n";
            return result;
        }

        cache[victim_line].block = static_cast<uint64_t>(block);
        result.evictions++;
    }

    return result;
}

Result baseline_simulate(const std::vector<uint64_t> &addresses, std::string policy) {
	std::vector<CacheLine> cache(CACHE_LINE);
	Result result;

	int cnt = 0;
	for (uint64_t address : addresses) {
		// check hit
		bool is_hit = false;
		uint64_t tag = address >> 6;

		for (CacheLine &c : cache) {
			if (c.valid && c.tag == tag) {
				is_hit = true;
				result.hits++;
				c.lastUsedAt = cnt;
				break;
			}
		}

		if (is_hit) {
			cnt++;
			continue;
		}

		result.misses++;

		// check available
		bool is_insert = false;
		for (CacheLine &c : cache) {
			if (!c.valid) {
				c.tag = tag;
				c.block = address / BLOCK_SIZE;
				c.valid = true;
				c.insertedAt = cnt;
				c.lastUsedAt = cnt;

				is_insert = true;
				break;
			}
		}

		if (is_insert) {
			cnt++;
			continue;
		}

		// replacement
		int victim = 0;

		for (int i = 1; i < CACHE_LINE; i++) {
			if (policy == "FIFO") {
				if (cache[i].insertedAt < cache[victim].insertedAt) {
					victim = i;
				}
			}else if (policy == "LRU") {
				if (cache[i].lastUsedAt < cache[victim].lastUsedAt) {
					victim = i;
				}
			} 
		}

		cache[victim].tag = tag;
		cache[victim].block = address / BLOCK_SIZE;
		cache[victim].valid = true;
		cache[victim].insertedAt = cnt;
		cache[victim].lastUsedAt = cnt;

		result.evictions++;
		cnt++;
	}
	return result;
}

void print_result(const std::string& policy,const Result& result,size_t access_count) {
    double hit_rate = 0.0;

    if (access_count > 0) {hit_rate = 100.0 * result.hits / access_count;
    }

    std::cout << policy
              << " | Hits: " << result.hits
              << " | Misses: " << result.misses
              << " | Evictions: " << result.evictions
              << " | Hit Rate: " << hit_rate << "%\n";
}

bool validate_workload(const std::string& name,
                       const std::vector<uint64_t>& addresses) {
    for (uint64_t address : addresses) {
        const uint64_t block = address / BLOCK_SIZE;
        if (block >= static_cast<uint64_t>(NUM_BLOCKS)) {
            std::cerr << "Unsupported block " << block << " in " << name
                      << ". This Q-table supports block IDs 0.."
                      << (NUM_BLOCKS - 1) << ".\n";
            return false;
        }
    }
    return true;
}

void run_workload(const std::string& name,
    const std::vector<uint64_t>& addresses) {
    std::cout << "\n=== " << name << " ===\n";
    std::cout << "Number of accesses: " << addresses.size() << '\n';

    const Result fifo = baseline_simulate(addresses, "FIFO");
    const Result lru = baseline_simulate(addresses, "LRU");
    const Result qlearning = simulate_q_learning(addresses);

    print_result("FIFO", fifo, addresses.size());
    print_result("LRU", lru, addresses.size());
    print_result("Q-Learning", qlearning, addresses.size());
}

int main() {
    // Training data: blocks 0, 1, and 2 recur frequently; blocks 3..6 are
    // introduced as less-frequent blocks. Addresses are byte addresses.
    const std::vector<uint64_t> training_workloads = {
    	0, 64, 128, 192,
         0, 64, 128, 256,
         0, 64, 128, 320,
         0, 64, 128, 384,
         0, 64, 128};

    // Test workloads are separate from training data. All blocks remain in
    // the supported range 0..6 so the Q-table state/action space is valid.
    const std::vector<uint64_t> test_hot_set = {
        0, 64, 128, 192, 0, 64, 128, 256,
        0, 64, 128, 320, 0, 64, 128, 384,
        0, 64, 128, 256, 0, 64, 128, 192
    };

    const std::vector<uint64_t> test_cyclic = {
        0, 64, 128, 192, 256, 320, 384,
        0, 64, 128, 192, 256, 320, 384,
        0, 64, 128, 192, 256, 320, 384
    };

    const std::vector<uint64_t> test_mixed = {
        0, 64, 128, 192,
        0, 64, 128, 256,
        192, 0, 64, 128,
        256, 0, 64, 128,
        320, 0, 64, 128,
        192, 0, 64, 128
    };

    if (!validate_workload("training workload", training_workloads) ||
        !validate_workload("test_hot_set", test_hot_set) ||
        !validate_workload("test_cyclic", test_cyclic) ||
        !validate_workload("test_mixed", test_mixed)) {
        return 1;
    }

    train_q_learning(training_workloads);

    std::cout << "Cache configuration: fully associative, " << CACHE_LINE
              << " lines, " << BLOCK_SIZE << " bytes per block\n";
    std::cout << "Q-learning training episodes: " << EPISODES << "\n";
    std::cout << "Note: Q-learning is trained on the training workloads above; results are workload-dependent.\n";

    run_workload("Test 1: Hot set with occasional blocks", test_hot_set);
    run_workload("Test 2: Cyclic access over 7 blocks", test_cyclic);
    run_workload("Test 3: Mixed locality", test_mixed);

    return 0;
}

