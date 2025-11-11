#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <atomic>
#include <thread>
#include <vector>

// Test concurrent access to getCurrentCUDABlasHandle and getCUDABlasLtWorkspace
// to verify that the data race fix is working correctly

TEST(CUDABlasHandlePoolTest, ConcurrentGetCurrentCUDABlasHandle) {
  if (!at::cuda::is_available()) {
    return;
  }

  constexpr int num_threads = 20;
  constexpr int iterations_per_thread = 100;

  std::atomic<int> error_count{0};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&error_count]() {
      try {
        at::cuda::CUDAGuard device_guard(0);

        for (int j = 0; j < iterations_per_thread; ++j) {
          // This should not crash or cause data races
          cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

          // Verify handle is valid (non-null)
          if (handle == nullptr) {
            error_count++;
          }
        }
      } catch (const std::exception& e) {
        error_count++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0);
}

TEST(CUDABlasHandlePoolTest, ConcurrentGetCUDABlasLtWorkspace) {
  if (!at::cuda::is_available()) {
    return;
  }

  constexpr int num_threads = 20;
  constexpr int iterations_per_thread = 100;

  std::atomic<int> error_count{0};
  std::vector<std::thread> threads;

  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&error_count]() {
      try {
        at::cuda::CUDAGuard device_guard(0);

        for (int j = 0; j < iterations_per_thread; ++j) {
          // This should not crash or cause data races
          void* workspace = at::cuda::getCUDABlasLtWorkspace();

          // Verify workspace is valid (non-null)
          if (workspace == nullptr) {
            error_count++;
          }
        }
      } catch (const std::exception& e) {
        error_count++;
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0);
}

TEST(CUDABlasHandlePoolTest, ConcurrentClearWorkspaces) {
  if (!at::cuda::is_available()) {
    return;
  }

  constexpr int num_accessor_threads = 15;
  constexpr int num_clear_threads = 5;
  constexpr int iterations_per_thread = 50;

  std::atomic<bool> stop{false};
  std::atomic<int> error_count{0};
  std::vector<std::thread> threads;
  threads.reserve(num_accessor_threads + num_clear_threads);

  // Launch accessor threads
  for (int i = 0; i < num_accessor_threads; ++i) {
    threads.emplace_back([&stop, &error_count]() {
      try {
        at::cuda::CUDAGuard device_guard(0);

        while (!stop.load(std::memory_order_relaxed)) {
          cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
          void* workspace = at::cuda::getCUDABlasLtWorkspace();

          if (handle == nullptr || workspace == nullptr) {
            error_count++;
          }
        }
      } catch (const std::exception& e) {
        error_count++;
      }
    });
  }

  // Launch threads that clear workspaces
  for (int i = 0; i < num_clear_threads; ++i) {
    threads.emplace_back([&error_count]() {
      try {
        for (int j = 0; j < iterations_per_thread; ++j) {
          at::cuda::clearCublasWorkspaces();
          std::this_thread::yield();
        }
      } catch (const std::exception& e) {
        error_count++;
      }
    });
  }

  // Let them run for a bit
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  stop.store(true, std::memory_order_relaxed);

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  c10::cuda::CUDACachingAllocator::init(1);
  return RUN_ALL_TESTS();
}
