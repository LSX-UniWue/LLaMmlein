import multiprocessing
import time


def logger_process(log_queue, log_file):
    try:
        with open(log_file, "a", buffering=4096) as f:
            while True:
                message = log_queue.get(timeout=5)
                if message == "STOP":
                    break
                f.write(message + "\n")
                f.flush()
    except Exception as e:
        print(f"Logger process encountered an error: {e}")


def monitor_logger_process(log_queue, log_file, interval=5):
    logger = multiprocessing.Process(target=logger_process, args=(log_queue, log_file))
    logger.start()

    while True:
        time.sleep(interval)  # Periodic check
        if not logger.is_alive():
            print("Logger process died. Restarting...")
            logger.join()  # Ensure the dead process is cleaned up
            logger = multiprocessing.Process(target=logger_process, args=(log_queue, log_file))
            logger.start()


def main_thread(log_queue):
    for i in range(10):
        log_queue.put(f"Main thread log {i}")
        time.sleep(0.5)


def dataloader_thread(log_queue):
    for i in range(10):
        log_queue.put(f"Dataloader thread log {i}")
        time.sleep(1)


if __name__ == "__main__":
    log_queue = multiprocessing.Queue()
    log_file = "training.log"

    monitor_thread = multiprocessing.Process(target=monitor_logger_process, args=(log_queue, log_file))
    monitor_thread.start()

    try:
        from threading import Thread

        t1 = Thread(target=main_thread, args=(log_queue,))
        t2 = Thread(target=dataloader_thread, args=(log_queue,))

        t1.start()
        t2.start()

        t1.join()
        t2.join()
    finally:
        log_queue.put("STOP")
        monitor_thread.terminate()
        monitor_thread.join()
