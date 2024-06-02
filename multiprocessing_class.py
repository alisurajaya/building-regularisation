import multiprocessing

class MultiProcessing:
    def __init__(self, num_processes=None) -> None:
        multiprocessing.freeze_support()
        self.num_processes = num_processes or multiprocessing.cpu_count()
    
    def process_data(self, func, args_lists):
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            try:
                results = pool.starmap(func, args_lists)
            except KeyboardInterrupt:
                pool.terminate()
        return results
    
    def close_pool(self):
        pass
