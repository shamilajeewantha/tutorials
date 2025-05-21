# With the multi-threaded version, you ceded control over task scheduling to the operating system, so the final order seemed random.

# runs on 5 threads
# uses single cpu

import threading
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import requests

# an object that resembles a global variable but is specific to each individual thread.
thread_local = threading.local()

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
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_site, sites)

def download_site(url):
    session = get_session_for_thread()
    with session.get(url) as response:
        thread_id = threading.get_ident()
        try:
            cpu_core = psutil.Process().cpu_num()
        except AttributeError:
            cpu_core = "N/A"  # Not supported on Windows/macOS
        print(f"[Thread {thread_id} | CPU {cpu_core}] Read {len(response.content)} bytes from {url}")

def get_session_for_thread():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

if __name__ == "__main__":
    main()