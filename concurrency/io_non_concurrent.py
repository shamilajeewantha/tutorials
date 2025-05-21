# In the synchronous version, you cycled through a list of sites and kept downloading their content in a deterministic order. 

# runs on single thread
# uses multiple cpus

import time

import requests
import psutil
import threading

def main():
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.perf_counter()
    download_all_sites(sites)
    duration = time.perf_counter() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")

def download_all_sites(sites):
    with requests.Session() as session:
        for url in sites:
            download_site(url, session)

def download_site(url, session):
    with session.get(url) as response:
        thread_id = threading.get_ident()
        try:
            cpu_core = psutil.Process().cpu_num()
        except AttributeError:
            cpu_core = "N/A"  # Not supported on Windows/macOS
        print(f"[Thread {thread_id} | CPU {cpu_core}] Read {len(response.content)} bytes from {url}")

if __name__ == "__main__":
    main()