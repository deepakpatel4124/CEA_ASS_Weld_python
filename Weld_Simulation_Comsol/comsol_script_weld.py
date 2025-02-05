import mph
import numpy as np
import time
import multiprocessing
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

#----------------------> Input Parameters

comsol_mph_path = r"Weld_201_domain_probe_83.mph"
transducer_fires = [0,31,63] #check for total firing element in the comsol model

save_name = "weld_velocity_test1"
directory_to_watch = r"C:\Users\Deepak\.comsol\v62\sessions"


#------------------------------------------------->
class LogFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.log_file = None

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".log"):
            if self.log_file:
                self.log_file.close()
            self.log_file = open(event.src_path, "r")
            print(f"Monitoring new log file: {event.src_path}")
            self.tail_file(self.log_file)

    def tail_file(self, file):
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.1)
                continue
            print(line, end="")


def monitor_directory(directory_path):
    event_handler = LogFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=directory_path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def run_comsol_model():

    # Start COMSOL client
    client = mph.Client(version="6.2")

    # Load the COMSOL model
    pymodel = client.load(comsol_mph_path)
    model = pymodel.java
    print("Running")

    # Iterate over theta values from 0 to 80 in steps of 10
    for transducer_fire in transducer_fires:
        
        # Setting the transducer firing element
        geom1_bnd = f"geom1_ls{286+transducer_fire}_bnd"
        model.component("comp1").physics("elte").feature("vel1").selection().named(geom1_bnd);
        # Run the study
        model.study("std1").run()
        # Save the changes with a unique filename for each theta
        filename_mph = f"{save_name}_{transducer_fire}.mph" 
        filename_table = f"{save_name}_{transducer_fire}.txt"
        filename_gif = f"{save_name}_{transducer_fire}.gif"

        try:
            # Export the table to a file
            model.result().export("tbl1").set("filename", filename_table)
            model.result().export("tbl1").run()

            # Export the animation to a GIF file
            model.result().export("anim1").set("target", "file")
            model.result().export("anim1").set("plotgroup", "pg3")
            model.result().export("anim1").set("giffilename", filename_gif)
            model.result().export("anim1").set("alwaysask", False)
            model.result().export("anim1").set("type", "movie")
            model.result().export("anim1").set("batch", False)
            model.result().export("anim1").run()
     

        except Exception as e:
            print(f"An error occurred during the export process: {e}")

        print(f"Finished for Transducer----->{transducer_fire}")

    print("All theta values processed.")


if __name__ == "__main__":
    # Start monitoring the directory in a separate process
    log_monitor_process = multiprocessing.Process(
        target=monitor_directory, args=(directory_to_watch,)
    )
    log_monitor_process.start()

    # Run the COMSOL model
    run_comsol_model()

    # Ensure the log monitor process stops when COMSOL finishes
    log_monitor_process.join()
