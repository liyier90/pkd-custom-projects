import csv
from datetime import datetime
from typing import Any, Dict, List


class CSVLogger:
    """A class to log the chosen information into csv dabble results"""

    def __init__(self, file_path, headers):
        headers.insert(0, "Time")

        self.csv_file = open(file_path, mode="a+")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=headers)

        # if file is empty write header
        if self.csv_file.tell() == 0:
            self.writer.writeheader()

        self.last_write = datetime.now()

    def write(self, data_pool: Dict[str, Any], specific_data: List[str]) -> None:
        content = {k: v for k, v in data_pool.items() if k in specific_data}
        curr_time = datetime.now()
        time_str = curr_time.strftime("%H:%M:%S")
        content.update({"Time": time_str})
        self.writer.writerow(content)

    def __del__(self):
        self.csv_file.close()
