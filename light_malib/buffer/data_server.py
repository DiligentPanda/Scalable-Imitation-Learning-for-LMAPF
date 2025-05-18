# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from .table import Table
from light_malib.utils.logger import Logger
import threading
from .database import Database
import copy

class DataServer:
    def __init__(self, id, cfg):
        self.id = id
        self.cfg = cfg
        self.tables = {}

        self.table_lock = threading.Lock()

        self.read_timeout = self.cfg.read_timeout
        self.table_cfg = self.cfg.table_cfg

        self.any_data = {}
        self.any_data_lock = threading.Lock()

        Logger.info("{} initialized".format(self.id))
        
    
    def put(self, key, data):
        with self.any_data_lock:
            self.any_data[key]=data
    
    def get(self, key, verbose=True): 
        try:
            with self.any_data_lock:      
                return self.any_data[key]
        except KeyError:
            if verbose:
                info = "{}::get key {} is not found".format(
                    self.id, key
                )
                Logger.warning(info)
            return None     
        
    def pop(self, key, verbose=True):
        try:
            with self.any_data_lock:      
                return self.any_data.pop(key)
        except KeyError:
            if verbose:
                info = "{}::pop key {} is not found".format(
                    self.id, key
                )
                Logger.warning(info)
            return None     

    def create_table(self, table_name, table_cfg=None):
        with self.table_lock:
            if table_cfg is None:
                table_cfg=self.table_cfg
            if table_name not in self.tables:
                Logger.warning("table_cfgs:{} uses {}".format(self.id, table_cfg))
                self.tables[table_name] = Table(table_cfg)
                Logger.info("{} created data table {}".format(self.id, table_name))
            else:
                Logger.warning("Table {} exists. Cannot overwrite it!".format(table_name))

    def remove_table(self, table_name):
        with self.table_lock:
            if table_name in self.tables:
                self.tables.pop(table_name)
            Logger.info("{} removed data table {}".format(self.id, table_name))

    def get_statistics(self, table_name):
        try:
            with self.table_lock:
                statistics = self.tables[table_name].get_statistics()
            return statistics
        except KeyError:
            time.sleep(1)
            info = "{}::get_table_stats: table {} is not found".format(
                self.id, table_name
            )
            Logger.warning(info)
            return {}

    def save(self, table_name, data):
        try:
            with self.table_lock:
                table: Table = self.tables[table_name]
            if len(data) > 0:
                table.write(data)
        except KeyError:
            time.sleep(1)
            Logger.warning(
                "{}::save_data: table for {} is not found".format(self.id, table_name)
            )

    def sample(self, table_name, batch_size, wait=False):
        try:
            with self.table_lock:
                table: Table = self.tables[table_name]
            samples = None
            samples = table.read(batch_size, timeout=self.read_timeout)
            if samples is not None:
                return samples, True
            else:
                return samples, False
        except KeyError:
            Logger.warning(
                "{}::sample_data: table {} is not found".format(self.id, table_name)
            )
            time.sleep(1)
            samples = None
            return samples, False

    def load_data(self, table_name, data_path):
        """
        TODO(jh): maybe support more data format?
        data now are supposed to be stored in pickle format as a list of samples.
        """
        # check extension
        assert data_path[-4:] == ".pkl"

        # get table
        with self.table_lock:
            table: Table = self.tables[table_name]

        # load data from disk
        import pickle as pkl

        with open(data_path, "rb") as f:
            samples = pkl.load(f)

        assert (
            len(samples) <= table.capacity
        ), "load too much data(size{}) to fit into table(capacity:{})".format(
            len(samples), table.capacity
        )

        # write samples to table
        table.write(samples)
        Logger.info(
            "Table {} load {} data from {}".format(self.id, len(samples), data_path)
        )

    def load_database(self, table_name, database_path):
        """
        load data from database to table
        """
        if isinstance(database_path,str):
            database_path=[database_path]
            
        for _database_path in database_path:
            database=Database(_database_path)

            # load data from database
            data = database.load_data()
            
            for (map_name, num_robots), _data in data.items():
                _table_name="{}_{}_{}".format(table_name, map_name, num_robots)
                
                Logger.info("Database load {} data from database".format(len(_data)))
            
                if _table_name not in self.tables:
                    table_cfg=copy.deepcopy(self.table_cfg)
                    table_cfg.update({"capacity":len(_data),"sampler_type":"uniform","sample_max_usage": 1e8, "rate_limiter_cfg":{}})
                    
                    # TODO(rivers): customize the table capactiy, etc.
                    self.create_table(_table_name, table_cfg)
                    
                table: Table = self.tables[_table_name]

                # write samples to table
                table.write(_data)
                
            Logger.info("Table for database {} is created".format(_database_path))