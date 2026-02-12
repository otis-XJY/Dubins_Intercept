import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调用转换后的函数
import pickle
import argparse
from intercept.IsoMap.WH_main_obtainMap import WH_main_obtainMapP,WH_main_obtainMapRef,WH_main_obtainIso
from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm
from Draw.Draw_pathA_no_circle_final_path import Draw_pathA_no_circle_final_path
from Draw.Draw_map import Draw_map
from intercept.IsoMap.obtainPath import obtainPath
from intercept.IsoMap.DrawIso import draw_iso


plt.figure(figsize=(10, 8))
Draw_map(Map['PStart_Point'], Map['Trans_Point'], Map['ValuePos'], Map['obs'], Map['sure'], Map['obs_no_circle'], Map['obs_no_circle_in'])
iii=0

pathEEEE=pathFinalTP2Val[ETP_idx[iii]][VP_idx[iii]]
plt.plot(pathEEEE[0,:int(Iso_idx[iii] - PathE2TP[iii].shape[1])], pathEEEE[1,:int(Iso_idx[iii] - PathE2TP[iii].shape[1])], 'r-', markersize=10)

for i in range(len(PathP2TP)):
    plt.plot(PathP2Iso[i][0,:], PathP2Iso[i][1,:], 'r-', markersize=10)
    plt.plot(PathIso2TP[i][0,:], PathIso2TP[i][1,:], 'b-', markersize=10)
# draw_iso(Map, path_segments, pathFinalP, vthetaAllP, interactive=1)

# plt.plot(Iso2TP_IsoPos[tpid][t][0,:],Iso2TP_IsoPos[tpid][t][1,:],'^')
plt.plot(pathFinalMapPTP2Iso[21][4][0,-200],pathFinalMapPTP2Iso[21][4][1,-200],'^')


tpid=13
tpidfrom=22
for t in range(num_Times):
    if len(Iso2TP_IsoPos[tpid][t])>1:
        plt.plot(Iso2TP_IsoPos[tpid][t][0,:],Iso2TP_IsoPos[tpid][t][1,:],'^')

plt.plot(pathFinalMapPTP2Iso[21][4][0,:],pathFinalMapPTP2Iso[21][4][1,:],'-')






# pathLLL=pathFinal[j][i]
# idid=len(pathLLL[0, :]) - round( Map['v_E'] * Map['timePlot'][tt])
# posiddd=- round( Map['v_E'] * Map['timePlot'][tt])
# plt.plot(pathLLL[0,:], pathLLL[1,:], 'k-', markersize=10)

# plt.plot(pathLLL[0,idid:], pathLLL[1,idid:], 'k-', markersize=10)
# plt.plot(pathLLL[0,posiddd], pathLLL[1,posiddd], '^', markersize=10)

# len(pathFinalMapPTP2Iso[tpidfrom][tpid][0, :]) - round(v_E * Map['timePlot'][0])
# Draw_pathA_no_circle_final_path(path_segments, 0,ax=None)

# for t in range(len(IsoMap_i_tt_P2Iso_raw[1])):
#     if len(IsoMap_i_tt_P2Iso_raw[1][t].IsoPos)>1:
#         plt.plot(IsoMap_i_tt_P2Iso_raw[1][t].IsoPos[0,:],IsoMap_i_tt_P2Iso_raw[1][t].IsoPos[1,:],'^')




plt.show()
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 8))
# plt.plot(TPIso_IsoPos_sub[0], TPIso_IsoPos_sub[1], '^', markersize=10)
# plt.show()


# PathE2Val_true 永远0-1-2

# 拼接后对应的是 IsoMap_i_tt_E(P)2Iso
# 拼接后的te 拼接后的tp EfromTPid  PfromTPid 
#         0          1         2         3
# IsoIdE IsoIdP IsoDist [IsoAngle IsoE2Vdist   
#      4      5       6 [       7
# Pos2IsoLen(整个拦截) cost ValId PtoTPid
#                   7    8     9      10
# Eid Pid 拼接后Eiso在PathId 拼接后Piso在PathId 
#  11  12                13                14
# CoseAll
# 15-17
pairsE2Val111_=np.array(    [[0,1],[1,2],[0,0]])
UnCapEid111=np.array([0,2])
pairsE2Val11111 = pairsE2Val111_[UnCapEid111, :]

UnCapPid=[i for i in range(3)]
np.where(1==UnCapPid)