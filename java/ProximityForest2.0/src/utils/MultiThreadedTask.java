package utils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MultiThreadedTask {
    /**
     * Stores some of the multi-threading tasks for PF and TS-CHIEF
     */
    private ExecutorService executor;

    public MultiThreadedTask(int numThreads) {
        numThreads = Math.min(numThreads, Runtime.getRuntime().availableProcessors());
        setExecutor(Executors.newFixedThreadPool(numThreads));
    }

    public ExecutorService getExecutor() {
        return executor;
    }

    public void setExecutor(ExecutorService executor) {
        this.executor = executor;
    }

    public static <T> ArrayList invokeParallelTasks(List<Callable<T>> tasks, MultiThreadedTask parallelTasks) throws Exception {
        List<Future<T>> results = parallelTasks.getExecutor().invokeAll(tasks);
        ArrayList<T> output = new ArrayList<>();
        for (Future<T> future : results) {
            output.add(future.get()); // block until complete
        }
        return output;
    }
}
