import concurrent.futures
import time

def do_something(sec1, sec2):
    time.sleep(sec1*sec2)
    print(f'waiting {sec1*sec2}')

if __name__ == "__main__":

    start = time.perf_counter()
    ### pass multiple arguments
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs1 = [5,4,3]
        secs2 = [1,4,3]
        results = executor.map(do_something, secs1, secs2)

    finish = time.perf_counter()    
    print(f'finished in {round(finish-start,2)} seconds')