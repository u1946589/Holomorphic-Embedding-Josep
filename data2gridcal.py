import pandas as pd
from GridCal.Engine import *


fname = '/home/santi/Documentos/GitHub/Holomorphic-Embedding-Josep/Data.xlsx'

bus_df = pd.read_excel(fname, sheet_name='Busos')
"""
Bus	P	Q	V	delta	Tipus
"""


grid = MultiCircuit()

# add buses
bus_dict = dict()
for i, row in bus_df.iterrows():
    idx = int(row['Bus'])
    bus = Bus(name='Bus_' + str(idx))
    tpe = row['Tipus']
    bus_dict[idx] = bus
    if tpe == 'PQ':
        load = Load(P=-row['P']*100, Q=-row['Q']*100)
        bus.add_device(load)
    elif tpe == 'PV':
        gen = Generator(active_power=row['P']*100, voltage_module=row['V'])
        bus.add_device(gen)
    elif tpe == 'Slack':
        bus.is_slack = True

    grid.add_bus(bus)


branch_df = pd.read_excel(fname, sheet_name='Topologia')
"""
Bus inici	Bus fi	R	X	B/2
"""
for i, row in branch_df.iterrows():
    f = int(row['Bus inici'])
    t = int(row['Bus fi'])
    b1 = bus_dict[f]
    b2 = bus_dict[t]
    branch = Branch(bus_from=b1, bus_to=b2, r=row['R'], x=row['X'], b=2*row['B/2'])
    grid.add_branch(branch)

FileSave(circuit=grid, file_name='/home/santi/Documentos/GitHub/Holomorphic-Embedding-Josep/helm_data1.gridcal').save()