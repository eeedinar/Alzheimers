import multiprocessing
import time 

start = time.perf_counter()

def do_something(sec1, sec2):
    time.sleep(sec1*sec2)
    print(f'waiting {sec1*sec2}')

### start processes in for loop
processes = []
for _ in range(10):    
    p = multiprocessing.Process(target = do_something, args = [1,2])
    p.start()
    processes.append(p)

### join processes started in for loop
for process in processes:
    process.join()

finish = time.perf_counter()    
print(f'finished in {round(finish-start,2)} seconds')
