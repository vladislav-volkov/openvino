#!/usr/bin/env python3

import argparse, csv
from pathlib import Path

Domain = ['CC0MKLDNNPlugin',
          'CC1MKLDNNPlugin',
          'CC2MKLDNNPlugin']

FILE_HEADER = "#pragma once\n\n"

FILE_FOOTER = "\n"

ENABLED_SCOPE_FMT = "#define CC0MKLDNNPlugin_%s 1\n"
ENABLED_SWITCH_FMT = "#define CC1MKLDNNPlugin_%s 1\n#define CC1MKLDNNPlugin_%s_cases %s\n"
ENABLED_FACTORY_INSTANCE_FMT = "#define CC2MKLDNNPlugin_%s 1\n"

class Scope:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def generate(self, f):
        f.write(ENABLED_SCOPE_FMT % self.name)

class Switch:
    def __init__(self, name):
        self.name = name
        self.cases = set()

    def case(self, val):
        self.cases.add(val)

    def generate(self, f):
        f.write(ENABLED_SWITCH_FMT % (self.name, self.name, ', '.join(self.cases)))

class Factory:
    def __init__(self, name):
        self.name = name
        self.registered = {}
        self.created = set()

    def register(self, id, name):
        self.registered[id] = name

    def create(self, id):
        self.created.add(id)

    def generate(self, f):
        for id in self.created:
            r = self.registered.get(id)
            f.write(ENABLED_FACTORY_INSTANCE_FMT % r)

class Stat:
    def __init__(self, files):
        self.scopes = set()
        self.switches = {}
        self.factories = {}
        self.read(files)

    def factory(self, name):
        if name not in self.factories:
            self.factories[name] = Factory(name)
        return self.factories.get(name)

    def switch(self, name):
        if name not in self.switches:
            self.switches[name] = Switch(name)
        return self.switches.get(name)

    def read(self, files):
        for stat in files:
            with open(stat) as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    # Scopes
                    scopes = list(filter(lambda row: row[0] == Domain[0], rows))
                    for row in scopes:
                        self.scopes.add(Scope(row[1]))

                    # Switches
                    switches = list(map(lambda row: row[1].strip().split('$'),
                                        filter(lambda row: row[0] == Domain[1], rows)))
                    for switch in switches:
                        self.switch(switch[0]).case(switch[1])

                    # Factories
                    factories = list(map(lambda row: row[1].strip().split('$'),
                                        filter(lambda row: row[0] == Domain[2], rows)))
                    for reg in list(filter(lambda row: row[0] == 'REG', factories)):
                        self.factory(reg[1]).register(reg[2], reg[3])
                    for cre in list(filter(lambda row: row[0] == 'CREATE', factories)):
                        self.factory(cre[1]).create(cre[2])

    def generate(self, out):
        with open(out, 'w') as f:
            f.write(FILE_HEADER)

            for scope in self.scopes:
                scope.generate(f)
            if self.scopes:
                f.write("\n")

            for _, switch in self.switches.items():
                switch.generate(f)
            if self.switches:
                f.write("\n")

            for _, factory in self.factories.items():
                factory.generate(f)

            f.write(FILE_FOOTER)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stat', type=Path, nargs='+', metavar='PATH[ PATH...]',
        help='IntelSEAPI statistics files in CSV format', required=True)
    parser.add_argument('--out', type=Path, metavar='cc.h',
        help='C++ header file to be generated', required=True)
    args = parser.parse_args()

    stat = Stat(args.stat)
    stat.generate(args.out)

if __name__ == '__main__':
    main()
