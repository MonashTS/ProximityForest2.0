#pragma once

#include "concepts.hpp"
#include "utils/uncopyable.hpp"
#include "utils/stats.hpp"

#include <any>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>

namespace libtempo::utils {

  /// Pick a random item from a subscriptable type, from [0] to [size-1]
  template<typename PRNG>
  inline const auto& pick_one(const Subscriptable auto& collection, size_t size, PRNG& prng) {
    if (size==1) { return collection[0]; }
    else if (size>1) {
      auto distribution = std::uniform_int_distribution<size_t>(0, size - 1);
      return collection[distribution(prng)];
    } else {
      throw std::invalid_argument("Picking from an empty collection");
    }
  }

  /// Pick a random item for a vector like type.
  template<typename VecLike, typename PRNG>
  inline const auto& pick_one(const VecLike& v, PRNG& prng) {
    return pick_one(v, v.size(), prng);
  }


  // --- --- --- --- --- ---
  // --- Constants
  // --- --- --- --- --- ---

  /// Constant to be use when no window is required
  constexpr size_t NO_WINDOW{std::numeric_limits<size_t>::max()};

  /// Positive infinity for float types
  template<typename FloatType>
  constexpr FloatType PINF{std::numeric_limits<FloatType>::infinity()};

  /// Negative infinity for float types
  template<typename FloatType>
  constexpr FloatType NINF{-PINF<FloatType>};

  /// Not A Number
  template<typename FloatType>
  constexpr FloatType QNAN{std::numeric_limits<FloatType>::quiet_NaN()};

  /// Lower Bound inital value, use to deal with numerical instability
  template<typename FloatType>
  FloatType INITLB{-pow(FloatType(10), -(std::numeric_limits<FloatType>::digits10 - 1))};



  // --- --- --- --- --- ---
  // --- Simple Tooling
  // --- --- --- --- --- ---

  /// Minimum of 3 values using std::min<T>
  template<typename T>
  inline T min(T a, T b, T c) { return std::min<T>(a, std::min<T>(b, c)); }

  /// Maximum of 3 values using std::min<T>
  template<typename T>
  inline T max(T a, T b, T c) { return std::max<T>(a, std::max<T>(b, c)); }



  // --- --- --- --- --- ---
  // --- Should not happen
  // --- --- --- --- --- ---

  /// Throw an exception "should not happen". Used as default case in switches.
  void inline should_not_happen() { throw std::logic_error("Should not happen"); }



  // --- --- --- --- --- ---
  // --- Unsigned tooling
  // --- --- --- --- --- ---


  /// Unsigned arithmetic: Given an 'index' and a 'window',
  /// get the start index corresponding to std::max(0, index-window)
  inline size_t cap_start_index_to_window(size_t index, size_t window) {
    if (index>window) { return index - window; } else { return 0; }
  }

  /// Unsigned arithmetic:
  ///  Given an 'index', a 'window' and an 'end', get the stop index corresponding to std::min(end, index+window+1).
  ///  The expression index+window+1 is illegal for any index>0 as window could be MAX-1
  inline size_t cap_stop_index_to_window_or_end(size_t index, size_t window, size_t end) {
    // end-window is valid when window<end
    if (window<end&&index + 1<end - window) { return index + window + 1; } else { return end; }
  }

  /// Absolute value for any comparable and subtractive type, without overflowing risk for unsigned types.
  template<std::unsigned_integral T>
  inline T absdiff(T a, T b) { return (a>b) ? a - b : b - a; }

  /// From unsigned to signed for integral types
  template<typename UIType>
  inline typename std::make_signed_t<UIType> to_signed(UIType ui) {
    static_assert(std::is_unsigned_v<UIType>, "Template parameter must be an unsigned type");
    using SIType = std::make_signed_t<UIType>;
    if (ui>(UIType)(std::numeric_limits<SIType>::max())) {
      throw std::overflow_error("Cannot store unsigned type in signed type.");
    }
    return (SIType)ui;
  }


  // --- --- --- --- --- ---
  // --- Initialisation tool
  // --- --- --- --- --- ---

  namespace initBlock_detail {
    struct tag {};

    template<class F>
    decltype(auto) operator +(tag, F&& f) {
      return std::forward<F>(f)();
    }
  }

#define initBlock initBlock_detail::tag{} + [&]() -> decltype(auto)

#define initBlockStatic initBlock_detail::tag{} + []() -> decltype(auto)


  // --- --- --- --- --- ---
  // --- Capsule
  // --- --- --- --- --- ---

  /// Capsule
  using Capsule = std::shared_ptr<std::any>;

  /// Capsule builder helper
  template<typename T, typename... Args>
  inline Capsule make_capsule(Args&& ... args) {
    return std::make_shared<std::any>(std::make_any<T>(args...));
  }

  /// Capsule pointer accessor
  template<typename T>
  inline T *get_capsule_ptr(const std::shared_ptr<std::any>& capsule) {
    return std::any_cast<T>(capsule.get());
  }

}

// --- --- --- --- --- ---
// --- Timing
// --- --- --- --- --- ---

namespace libtempo::utils {

  using myclock_t = std::chrono::steady_clock;
  using duration_t = myclock_t::duration;
  using time_point_t = myclock_t::time_point;

  /** Create a time point for "now" */
  time_point_t now();

  /** Print a duration in a human readable form (from nanoseconds to hours) in an output stream. */
  void printDuration(std::ostream& out, const duration_t& elapsed);

  /** Shortcut to print in a string */
  std::string as_string(const duration_t& elapsed);

  /** Shortcut for the above function, converting two time points into a duration. */
  void printExecutionTime(std::ostream& out, time_point_t start_time, time_point_t end_time);

  /** Shortcut to print in a string */
  std::string as_string(time_point_t start_time, time_point_t end_time);

} // End of namespace libtempo::utils::timing


// --- --- --- --- --- ---
// --- Parallel tasks
// --- --- --- --- --- ---

namespace libtempo::utils {

  /// Helper class to execute several tasks in parallel.
  /// Tasks must be prepared (with push_task) before being executed.
  /// The 'execute' method waits for all task to be completed.
  /// If the number of thread required is <= 1, the current thread is used.
  /// Else, the requested number of threads are spawned, and the current thread waits for their completion.
  class ParTasks {

  public:
    using task_t = std::function<void()>;
    using taskgen_t = std::function<std::optional<task_t>()>;

  private:
    std::mutex mtx;
    std::vector<std::thread> threads;
    std::queue<task_t> tasklist;

  public:

    ParTasks() = default;

    /// Non thread safe! Add all the task before calling "execute"
    void push_task(task_t func);

    /// Template version
    template<class F, class... Args>
    void push_task(F&& f, Args&& ... args) {
      tasklist.emplace(std::move(std::bind(std::forward<F>(f), std::forward<Args...>(args...))));
    }

    /// Blocking call
    void execute(int nbthreads);

    /// Blocking call
    void execute(int nbthreads, int nbtask);

    /// Blocking call using a task generator
    void execute(int nbthread, taskgen_t tgenerator);

  private:

    void run_thread();

    void run_thread(size_t nbtask);

    void run_thread_generator(taskgen_t& tgenerator);

  };

} // end of namespace libtempo::utils


// --- --- --- --- --- ---
// --- Progress Monitor
// --- --- --- --- --- ---

namespace libtempo::utils {

  struct ProgressMonitor {

    size_t total;

    explicit ProgressMonitor(size_t max);

    /// Print progress
    void print_progress(std::ostream& out, size_t nbdone);
  };

}