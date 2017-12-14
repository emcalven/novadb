#!/usr/bin/python
import sys
sys.path.append("../")
import time
import getopt
import logging
from nova.io import IO, ImportExport
import tarfile
import os
from glob import glob
import datetime


def make_tarfile(output_filename, source_files, model_path):
    with tarfile.open(output_filename, "w:bz2") as tar:
        for name in source_files:
            model_files = glob(model_path + name + ".*")
            for model_file in model_files:
                tar.add(model_file)
        #tar.add(source_dir, arcname=os.path.basename(source_dir))


class Daemon(object):
    def __init__(self, model_path="", db_path="../db/default"):
        self.model_path = model_path
        self.db_path = db_path
        self.archived_models = []
        self.new_models = None
        self.is_archived = False

        if not self.db_path.endswith(".db"):
            self.db_path += ".db"
        self.io = IO(db=self.db_path)
        self.io.connect_to_db()
        self.db_engine = self.io.engine
        self.ie = ImportExport(engine=self.db_engine)

    def run(self, model_path=None, **kwargs):
        if model_path is None:
            model_path = self.model_path

        export = self.ie.load_models(path=model_path, read_grains=True)
        if export:
            self.ie.to_sql(**kwargs)

    def archive_models(self, nmin=1000):
        self.new_models = list(set(self.ie.exported_models).difference(self.archived_models))
        if len(self.new_models) >= nmin:
            make_tarfile(self.model_path + str(datetime.datetime.now()).split('.')[0].replace(" ", "@"),
                         self.new_models, self.model_path)
            self.is_archived = True
        else:
            self.is_archived = False

    def delete_models(self):
        if self.is_archived:
            for name in self.new_models:
                model_files = glob(self.model_path + name + ".*")
                for model_file in model_files:
                    if os.path.isfile(model_file):
                        os.remove(model_file)



def main(argv):
    db_path = ''
    model_path = ''
    #try:
    opts, args = getopt.getopt(argv, "hd:m:", ["db_path=", "model_path="])
    #except getopt.GetoptError:
    #    print('db_daemon.py -d <db_path> -m <model_path>')
    #    sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -d <db_path> -m <model_path>')
            sys.exit()
        elif opt in ("-d", "--db_path"):
            db_path = arg
        elif opt in ("-m", "--model_path"):
            model_path = arg
    print('Database path is ', db_path)
    print('Model path is ', model_path)

    logger.info("Starting daemon...")
    print("Starting daemon...")
    daemon = Daemon(model_path=model_path, db_path=db_path)
    logger.info("Daemon running...")
    print("\nDaemon running...")

    try:
        while True:
            daemon.run()
            daemon.archive_models(nmin=3)
            daemon.delete_models()
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Stopping daemon...")
        print("Stopping daemon...")

if __name__ == "__main__":
    # Create logger
    logger = logging.getLogger('NOVA')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('daemon.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("\n==================================================================\n"
                "\n DAEMON STARTED"
                "\n==================================================================")
    main(sys.argv[1:])