# While the asynchronous version may show some clustering of completions, it’s generally non-deterministic due to changing network conditions.

# runs on the same thread
# uses multiple cpus

import asyncio
import time

import aiohttp
import requests

import psutil
import threading


async def main():
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.perf_counter()
    await download_all_sites(sites)
    duration = time.perf_counter() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")

async def download_all_sites(sites):
    # this gets stuck at gather because there is only one threads and all download tasks block it.
    # with requests.Session() as session:
    # The tasks can share the session because they’re all running on the same thread.     
    async with aiohttp.ClientSession() as session:
        # Notice that you don’t await the individual coroutine objects, as doing so would lead to executing them sequentially.
        tasks = [download_site(url, session) for url in sites]
        await asyncio.gather(*tasks, return_exceptions=True)

async def download_site(url, session):
    async with session.get(url) as response:
        thread_id = threading.get_ident()
        try:
            cpu_core = psutil.Process().cpu_num()
        except AttributeError:
            cpu_core = "N/A"  # Not supported on Windows/macOS
        print(f"[Thread {thread_id} | CPU {cpu_core}] Read {len(await response.read())} bytes from {url}")

if __name__ == "__main__":
    asyncio.run(main())