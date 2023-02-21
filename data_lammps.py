import numpy as np
import math
import re
import os
import random


class BasicCal(object):
    @staticmethod
    def get_vector(coor1, coor2, box=100.0):
        tmp1 = coor2 - coor1 - box * np.floor((coor2 - coor1) / box)
        return tmp1 - round(tmp1 / box) * box

    def caldis(self, atom1, atom2, box, maxleng=3.5):
        tmp_arr = np.array([], dtype=np.float32)
        for ix in range(len(atom1)):
            tmpleng = np.abs(self.get_vector(atom1[ix], atom2[ix], box[ix]))
            if tmpleng > maxleng:
                return 0
            else:
                tmp_arr = np.append(tmp_arr, [np.float32(tmpleng)])
        finres = np.sqrt(tmp_arr.dot(tmp_arr))
        if finres <= maxleng:
            return finres
        else:
            return 0

    @staticmethod
    def caldis2(atom1, atom2, box, dis):
        length = dis
        halfx = box[0] / 2
        halfy = box[1] / 2
        halfz = box[2] / 2
        lenx = halfx - abs(halfx - abs(atom1[0] - atom2[0]))
        if lenx > length:
            return 0
        leny = halfy - abs(halfy - abs(atom1[1] - atom2[1]))
        if leny > length:
            return 0
        lenz = halfz - abs(halfz - abs(atom1[2] - atom2[2]))
        if lenz > length:
            return 0
        lenxyz = math.sqrt(lenx * lenx + leny * leny + lenz * lenz)
        if lenxyz < length:
            return lenxyz
        else:
            return 0


class FinalLmpData(BasicCal):
    def __init__(self, filename, add_z, add_y=[]):
        self.filename = filename
        self.atomlist = []
        self.bondlist = []
        self.anglist = []
        self.dihlist = []
        self.implist = []
        self.atomtype = []
        self.bondtype = []
        self.angtype = []
        self.dihtype = []
        self.imptype = []
        self.charge = []
        self.mass = {}
        self.paircoeff = {}
        self.bondcoeff = {}
        self.angcoeff = {}
        self.dihcoeff = {}
        self.impcoeff = {}

        self.set_coeff()

        self.molnum = []
        self.moltimes = []
        self.moltot = []
        self.molname = []
        self.molcharge = []
        self.molbond = []
        self.molang = []
        self.moldih = []
        self.molimp = []
        self.box = []
        self.addzlen = 10.0

        for layer in range(len(add_z)):
            print("deal with the %d layer:" % layer)
            self.readquartz(stsub=8, aosub=16, mgsub=0)
            print("quartz or clay has been read, now write the z.inp")
            self.writeinp(direction="z", addmol=add_z[layer])
            for ix in range(len(add_z[layer])):
                if os.path.exists(add_z[layer][ix][0] + '.data'):
                    self.trandata(add_z[layer][ix][0])
                elif os.path.exists(add_z[layer][ix][0] + '.xyz'):
                    self.single(add_z[layer][ix][0] + '.xyz', 1.6)
                else:
                    print("no such file:", add_z[layer][ix][0])
            for ix in range(len(add_z[layer])):
                self.molname.append(add_z[layer][ix][0])
                file2 = open(add_z[layer][ix][0] + '.xyz')
                self.molnum.append(len(file2.readlines()) - 2)
                file2.close()
                self.moltimes.append(add_z[layer][ix][3])
            print("now packmol:", end=' ')
            os.system('~/apps/packmol/packmol < z.inp > /dev/null')
            print("packmol has been done;", end=' ')
            self.readxyz("z.xyz")
            os.system('rm z.inp z.xyz')
            print("all atoms has been read;")
        for layer in range(len(add_y)):
            print("deal with the %d layer in y direction:" % layer)
            self.writeinp(direction="y", addmol=add_y[layer])
            for ix in range(len(add_y[layer])):
                if os.path.exists(add_y[layer][ix][0] + '.data'):
                    self.trandata(add_y[layer][ix][0])
                elif os.path.exists(add_y[layer][ix][0] + '.xyz'):
                    self.single(add_y[layer][ix][0] + '.xyz', 1.6)
                else:
                    print("no such file:", add_y[layer][ix][0])
            for ix in range(len(add_y[layer])):
                self.molname.append(add_y[layer][ix][0])
                file2 = open(add_y[layer][ix][0] + '.xyz')
                self.molnum.append(len(file2.readlines()) - 2)
                file2.close()
                self.moltimes.append(add_y[layer][ix][3])
            print("now packmol:", end=' ')
            os.system('~/apps/packmol/packmol < y.inp > /dev/null')
            print("packmol has been done;", end=' ')
            self.readxyz("y.xyz")
            os.system('rm y.inp y.xyz')
            print("all atoms has been read;")
        self.calid()
        self.addbond_ang()
        self.caltype()
        print("now write the finallmp")
        self.output('finallmp')

    @staticmethod
    def get_charge(ele_name):
        char_dic = {"st": 2.1000, "ao": 1.5750, "ob": -1.0500, "obts": -1.1688, "obos": -1.1808, "ohy": -0.9500,
                    "ohs": -1.0808, "mgo": 1.3600, "at": 1.5750, "hoy": 0.4250, "eow": -0.8476, "ehw": 0.4238,
                    "na": 1.0000, "ow": -0.8476, "hw": 0.4238, "emgo": 1.2292, "cl": -1.0000, "k": 1.0000,
                    "oc": -0.3256, "co": 0.6512, "HC": 0.060, "CT": -0.240, "ca": 2.00000, "mg": 2.0000,
                    "ba": 2.0000, "c3": -0.0300, "c2": -0.0200, "h": 0.0100, "O": -1.0500, "nax": 1.0000,
                    "li": 1.0000, "lix": 1.0000, "kx": 1.0000, "w_ow": -0.8476, "w_hw": 0.4238}  # charge dictionary
        # char_dic["ob"] = -1.1430 #comment for trioctahedral clay; if for dioctahedral clay donnot comment this line
        if ele_name in char_dic.keys():
            return char_dic[ele_name]
        else:
            print("can't find charge for element in function get_charege:", ele_name)
            return 0

    def set_coeff(self):
        self.bondcoeff = {"hoy-ohy": ['554.1300', '1.000'], "hw-ow": ['554.1300', '1.0000'], "w_hw-w_ow": ['554.1300', '1.0000'],
                          "eow-ehw": ['517.600', '1.000'], "CT-HC": ['331.0000', '1.0900'], "co-oc": ['2027.9254', '1.1620']}
        self.angcoeff = {"hw-ow-hw": ['harmonic', '33.43000', '109.47000'], "w_hw-w_ow-w_hw": ['harmonic', '33.43000', '109.47000'],
                         "ehw-eow-ehw": ["harmonic", '33.43000', '109.47000'],
                         "hoy-ohy-st": ['harmonic', '6.00000', '110.10000'],
                         "hoy-ohy-ao": ["harmonic", '15.00000', '110.10000'],
                         "hoy-ohy-mgo": ["harmonic", '15.00000', '110.10000'],
                         "hoy-ohs-ao": ["harmonic", '15.00000', '110.10000'], "oc-co-oc": ["harmonic", '108.0660', '180.0000'],
                         "hoy-ohs-mgo": ["harmonic", '6.00000', '110.10000'], "HC-CT-HC": ["harmonic", '35.0000', '109.500']}
        self.dihcoeff = {}
        self.mass = {"st": 28.08600, "at": 26.98150, "ao": 26.98150, "mgo": 24.30500, "emgo": 24.30500, "ob": 15.99940,
                     "obts": 15.99940, "obos": 15.99940, "ohy": 15.99940, "ohs": 15.99940, "hoy": 1.007940,
                     "eow": 15.99940, "ehw": 1.007940, "ow": 15.99940, "hw": 1.007940, "na": 22.989768,
                     "cl": 35.45000, "oc": 15.99940, "co": 12.00000, "HC": 1.007940, "CT": 12.000000,
                     "ca": 40.07980, "k": 39.0983, "ba": 137.327, "mg": 24.305060, "nax": 22.989768,
                     "kx": 39.0983, "li": 6.9410, "lix": 6.9410}  # mass dictionary
        self.paircoeff = {"st": [0.0000018405, 3.30203000], "at": [0.0000018405, 3.30203000],
                          "ao": [0.0000013298, 4.2712400], "mgo": [0.0000009030, 5.2643200],
                          "ob": [0.1554000000, 3.1655400], "obos": [0.1554000000, 3.1655400],
                          "obts": [0.1554000000, 3.1655400], "ohy": [0.1554000000, 3.1655400],
                          "ohs": [0.1554000000, 3.1655400], "hoy": [0.0000000000, 0.0000000],
                          "eow": [0.1553000000, 3.1660000], "ehw": [0.0000000000, 0.0000000],
                          "ow": [0.155300000, 3.1660000], "hw": [0.0000000000, 0.0000000],
                          "na": [0.1553000000, 2.2344000], "emgo": [0.0000009030, 5.2643200],
                          "cl": [0.1001000000, 4.3999620], "oc": [0.1645070000, 3.0280000],
                          "co": [0.0559270000, 2.8000000], "HC": [0.0000000000, 0.0000000],
                          "CT": [0.0660000000, 3.5000000], "ca": [0.0999999980, 2.8719902],
                          "k": [0.1553000000, 2.8944000], "ba": [0.047000, 3.816607],
                          "mg": [0.875000, 1.643515], "nax": [0.1301000000, 2.3500100],
                          "kx": [0.1000000, 3.33400700], "li": [0.1553000000, 1.4740000],
                          "lix": [0.1553000000, 1.4740000]}  # pair_coeff dictionary

    def readquartz(self, stsub=0, aosub=0, mgsub=0):
        miner_atomlist = []
        edge_wat = []
        file_qz = open(self.filename)
        self.box = [float(ix) for ix in file_qz.readline().split()]
        qzcontents = file_qz.readlines()
        file_qz.close()
        for ix in range(len(qzcontents)):
            tmp_split = qzcontents[ix].split()
            if tmp_split[0] not in ["eow", "ehw"]:
                miner_atomlist.append([tmp_split[0]] + [float(jx) for jx in tmp_split[1:]])
            else:
                edge_wat.append([tmp_split[0]] + [float(jx) for jx in tmp_split[1:]])
        self.substi(miner_atomlist, stsub=stsub, aosub=aosub, mgsub=mgsub, whe_edge=len(edge_wat))
        self.molnum.append(len(miner_atomlist))
        self.moltimes.append(1)
        self.molname.append('mineral_piece')

        addzlen = self.findang(miner_atomlist)
        for ix in range(len(edge_wat)):
            edge_wat[ix][3] += addzlen
        edgenum = self.cal_edgewat(edge_wat)
        self.molnum.append(3)
        self.moltimes.append(edgenum)
        self.molname.append('edge_wat')

    def cal_edgewat(self, edgewat):
        tranwat = []
        eow_num = 0

        for ix in range(len(edgewat)):
            if edgewat[ix][0] == "eow":
                eow_num += 1
                atom_1 = edgewat[ix][1:4]
                tranwat.append(edgewat[ix])
                for jx in range(len(edgewat)):
                    if edgewat[jx][0] == "ehw":
                        atom_2 = edgewat[jx][1:4]
                        if self.caldis(atom_1, atom_2, box=[500, 500, 500], maxleng=1.3):
                            tranwat.append(edgewat[jx])
        self.molcharge.append([self.get_charge("eow"), self.get_charge("ehw"), self.get_charge("ehw")])
        self.molbond.append([['eow-ehw', 1, 2], ['eow-ehw', 1, 3]])
        self.molang.append([['ehw-eow-ehw', 2, 1, 3]])
        self.moldih.append([])
        self.molimp.append([])
        for ix in range(len(edgewat)):
            self.atomlist.append(tranwat[ix])
        return eow_num

    def substi(self, claycoor, stsub=0, aosub=0, mgsub=0, whe_edge=0):
        siindex = []
        alindex = []
        mgindex = []
        alsub_ex = []
        sisub_ex = []
        mgsub_ex = []
        si_edge = []
        al_edge = []
        mg_edge = []
        si_y = []
        al_y = []
        mg_y = []
        for ix in range(len(claycoor)):
            if claycoor[ix][0] == "Al":
                claycoor[ix][0] = "ao"
                alindex.append(ix)
                al_y.append(claycoor[ix][2])
            elif claycoor[ix][0] == "Si":
                claycoor[ix][0] = "st"
                siindex.append(ix)
                si_y.append(claycoor[ix][2])
            elif claycoor[ix][0] == "Mg":
                claycoor[ix][0] = "mgo"
                mgindex.append(ix)
                mg_y.append(claycoor[ix][2])
            elif claycoor[ix][0] in ["O", "O1", "O2", "O3"]:
                claycoor[ix][0] = "ob"
            elif claycoor[ix][0] in ["OH", "O-H"]:
                claycoor[ix][0] = "ohy"
            elif claycoor[ix][0] in ["H"]:
                claycoor[ix][0] = "hoy"
        # find the edge cation in the clay
        if whe_edge:
            if len(siindex):
                si_ymin, si_ymax = sorted(si_y)[0], sorted(si_y)[-1]
                for ix in siindex:
                    if claycoor[ix][2] > (si_ymax - 1) or claycoor[ix][2] < (si_ymin + 1):
                        si_edge.append(ix)
            if len(alindex):
                al_ymin, al_ymax = sorted(al_y)[0], sorted(al_y)[-1]
                for ix in alindex:
                    if claycoor[ix][2] > (al_ymax - 1) or claycoor[ix][2] < (al_ymin + 1):
                        al_edge.append(ix)
            if len(mgindex):
                mg_ymin, mg_ymax = sorted(mg_y)[0], sorted(mg_y)[-1]
                for ix in mgindex:
                    if claycoor[ix][2] > (mg_ymax - 1) or claycoor[ix][2] < (mg_ymin + 1):
                        mg_edge.append(ix)

        for ix in range(len(claycoor)):
            if claycoor[ix][0] == "hoy":
                atom_1 = claycoor[ix][1:4]
                for jx in range(len(claycoor)):
                    if claycoor[jx][0] in ["ob"]:
                        atom_2 = claycoor[jx][1:4]
                        if self.caldis2(atom_1, atom_2, box=self.box, dis=1.3):
                            claycoor[jx][0] = "ohy"
        while len(alsub_ex) < aosub:
            sub_tmp = random.choice(alindex)
            judge = 1
            if sub_tmp in alsub_ex:
                continue
            if sub_tmp in al_edge:
                continue
            else:
                for jx in range(len(alsub_ex)):
                    atom_1 = claycoor[alsub_ex[jx]][1:4]
                    atom_2 = claycoor[sub_tmp][1:4]
                    if self.caldis(atom_1, atom_2, box=self.box):
                        judge = 0
                        break
            if judge:
                alsub_ex.append(sub_tmp)
                claycoor[sub_tmp][0] = "mgo"
        while len(sisub_ex) < stsub:
            sub_tmp = random.choice(siindex)
            judge = 1
            if sub_tmp in sisub_ex:
                continue
            if sub_tmp in si_edge:
                continue
            else:
                for jx in range(len(sisub_ex)):
                    atom_1 = claycoor[sisub_ex[jx]][1:4]
                    atom_2 = claycoor[sub_tmp][1:4]
                    if self.caldis(atom_1, atom_2, box=self.box):
                        judge = 0
                        break
                for jx in range(len(alsub_ex)):
                    atom_1 = claycoor[sub_tmp][1:4]
                    atom_2 = claycoor[alsub_ex[jx]][1:4]
                    if self.caldis(atom_1, atom_2, box=self.box):
                        judge = 0
                        break
            if judge:
                sisub_ex.append(sub_tmp)
                claycoor[sub_tmp][0] = "at"
        while len(mgsub_ex) < mgsub:
            sub_tmp = random.choice(mgindex)
            judge = 1
            if sub_tmp in mgsub_ex:
                continue
            if sub_tmp in mg_edge:
                continue
            else:
                for jx in range(len(mgsub_ex)):
                    atom_1 = claycoor[mgsub_ex[jx]][1:4]
                    atom_2 = claycoor[sub_tmp][1:4]
                    if self.caldis(atom_1, atom_2, box=self.box):
                        judge = 0
                        break
            if judge:
                mgsub_ex.append(sub_tmp)
                claycoor[sub_tmp][0] = "ao"

        for ix in sisub_ex:
            for jx in range(len(claycoor)):
                atom_1 = claycoor[ix][1:4]
                atom_2 = claycoor[jx][1:4]
                if atom_1 != atom_2 and self.caldis(atom_1, atom_2, box=self.box, maxleng=2.8):
                    if claycoor[jx][0] == "ob":
                        claycoor[jx][0] = "obts"
                    elif claycoor[jx][0] == "ohy":
                        claycoor[jx][0] = "ohs"
        for ix in alsub_ex:
            for jx in range(len(claycoor)):
                atom_1 = claycoor[ix][1:4]
                atom_2 = claycoor[jx][1:4]
                if atom_1 != atom_2 and self.caldis(atom_1, atom_2, box=self.box, maxleng=2.8):
                    if claycoor[jx][0] == "ob":
                        claycoor[jx][0] = "obos"
                    elif claycoor[jx][0] == "ohy":
                        claycoor[jx][0] = "ohs"
        for ix in mgsub_ex:
            for jx in range(len(claycoor)):
                atom_1 = claycoor[ix][1:4]
                atom_2 = claycoor[jx][1:4]
                if atom_1 != atom_2 and self.caldis(atom_1, atom_2, box=self.box, maxleng=2.6):
                    if claycoor[jx][0] == "ob":
                        claycoor[jx][0] = "obos"
                    elif claycoor[jx][0] == "ohy":
                        claycoor[jx][0] = "ohs"

    def findang(self, miner_atomlist):
        bond_p = []
        ang_p = []
        charge_p = []
        for ix in range(len(miner_atomlist)):
            if miner_atomlist[ix][0] in ["hoy"]:
                for jx in range(len(miner_atomlist)):
                    if miner_atomlist[jx][0] in ["ohy", "ohs"]:
                        tmp_1 = miner_atomlist[ix][1:4]
                        tmp_2 = miner_atomlist[jx][1:4]
                        if self.caldis2(tmp_1, tmp_2, self.box, 1.5):
                            bond_p.append(['hoy' + '-' + 'ohy', ix + 1, jx + 1])
                            for kx in range(len(miner_atomlist)):
                                if miner_atomlist[kx][0] in ["mgo", "ao", "st", "at"]:
                                    tmp_1 = miner_atomlist[jx][1:4]
                                    tmp_2 = miner_atomlist[kx][1:4]
                                    if self.caldis2(tmp_1, tmp_2, self.box, 2.4):
                                        ang_p.append([miner_atomlist[ix][0] + '-' + miner_atomlist[jx][0] + '-'
                                                      + miner_atomlist[kx][0], ix + 1, jx + 1, kx + 1])
        for kx in range(len(miner_atomlist)):
            charge_p.append(self.get_charge(miner_atomlist[kx][0]))
        add_perz = 0
        if len(self.atomlist):
            atomsz = [float(self.atomlist[tmp_z][3]) for tmp_z in range(len(self.atomlist))]
            max_atomz = max(atomsz)
            minerz = [float(miner_atomlist[tmp_z][3]) for tmp_z in range(len(miner_atomlist))]
            min_minerz = min(minerz)
            add_perz = max(max_atomz - min_minerz + 1.7, self.addzlen)
            for per_miner_atom in range(len(miner_atomlist)):
                miner_atomlist[per_miner_atom][3] += add_perz
        for kx in range(len(miner_atomlist)):
            self.atomlist.append(miner_atomlist[kx])
        self.molcharge.append(charge_p)
        self.molbond.append(bond_p)
        self.molang.append(ang_p)
        self.moldih.append([])
        self.molimp.append([])
        return add_perz

    def writeinp(self, addmol=[], direction="z"):
        # direction must be z or y, for z we insert the water into the interlayer, for y we insert the water
        # hanging up y parallel to the xz plane
        arr_xx = np.array(np.array(self.atomlist)[:, 1], dtype=float)
        xx_sort = np.sort(arr_xx)
        arr_yy = np.array(np.array(self.atomlist)[:, 2], dtype=float)
        yy_sort = np.sort(arr_yy)
        arr_zz = np.array(np.array(self.atomlist)[:, 3], dtype=float)
        zz_sort = np.sort(arr_zz)
        xlo = (xx_sort[0] + xx_sort[len(xx_sort) - 1]) / 2 - self.box[0] / 2
        xhi = (xx_sort[0] + xx_sort[len(xx_sort) - 1]) / 2 + self.box[0] / 2
        yhi = (yy_sort[0] + yy_sort[len(yy_sort) - 1]) / 2 + self.box[1] / 2
        if direction == "z":
            deltaz = 0
            for ix in range(len(addmol)):
                deltaz += addmol[ix][3] * addmol[ix][1] * 10 / \
                          (6.02 * (self.box[0] - 2) * (self.box[1] - 2) * addmol[ix][2])
            fin_xlo = xlo + 1
            fin_xhi = xhi - 1
            fin_ylo = yy_sort[0] + 1
            fin_yhi = yy_sort[-1] - 1
            fin_zlo = zz_sort[-1] + 1.7
            fin_zhi = zz_sort[-1] + 1.7 + deltaz
            self.box[2] = fin_zhi - zz_sort[0]
        elif direction == "y":
            deltay = 0
            for ix in range(len(addmol)):
                deltay += addmol[ix][3] * addmol[ix][1] * 10 / (6.02 * (self.box[0] - 2) * self.box[2] * addmol[ix][2])
            fin_xlo = xlo + 1
            fin_xhi = xhi - 1
            fin_ylo = yhi + 1
            fin_yhi = yhi + 1 + deltay
            fin_zlo = zz_sort[0] + 0.6
            fin_zhi = zz_sort[0] + self.box[2] - 0.6
            self.box[1] += deltay
        else:
            print("wrong direction in packmol input")
            return 0
        file2 = open(direction + ".inp", "w")
        print("tolerance 2.0", file=file2)
        print("output " + direction + ".xyz", file=file2)
        for ix in range(len(addmol)):
            print("filetype xyz", file=file2)
            print("structure " + addmol[ix][0] + ".xyz", file=file2)
            print("number ", addmol[ix][3], file=file2)
            print("inside box", fin_xlo, fin_ylo, fin_zlo, fin_xhi, fin_yhi, fin_zhi, file=file2)
            print("end structure", file=file2)
        file2.close()

    def readxyz(self, filenm):
        file1 = open(filenm)
        file1.readline()
        file1.readline()
        # content = file1.readlines()
        # file1.close()
        # for ix in range(len(content)):
        #     self.atomlist.append([content[ix].split()[0]] + [float(jx) for jx in content[ix].split()[1:]])
        txtline = file1.readline()
        while txtline:
            self.atomlist.append([txtline.split()[0]] + [float(jx) for jx in txtline.split()[1:]])
            txtline = file1.readline()

    def single(self, filenm, leng):
        atoms_t = []
        atomtype = []
        pair = []
        charge_p = []
        ang_p = []
        dih_p = []
        bond_num = {}  # in order to calculate the angle and dihedral
        file2 = open(filenm)
        file2.readline()
        file2.readline()
        content = file2.readlines()
        file2.close()
        for ix in content:
            tmp_1 = ix.split()[0]
            tmp_2 = float(ix.split()[1])
            tmp_3 = float(ix.split()[2])
            tmp_4 = float(ix.split()[3])
            atomtype.append(tmp_1)
            charge_p.append(self.get_charge(tmp_1))
            atoms_t.append([tmp_2, tmp_3, tmp_4])
        for ix in range(len(atoms_t)):
            for jx in range(ix + 1, len(atoms_t)):
                if self.caldis2(atoms_t[ix], atoms_t[jx], box=[100, 100, 100], dis=leng):
                    if atomtype[ix] + '-' + atomtype[jx] in self.bondcoeff:
                        pair.append([atomtype[ix] + '-' + atomtype[jx], ix, jx])
                    elif atomtype[jx] + '-' + atomtype[ix] in self.bondcoeff:
                        pair.append([atomtype[jx] + '-' + atomtype[ix], jx, ix])
                    else:
                        print("no such bond coeff:", atomtype[ix] + '-' + atomtype[jx])
        for ix in range(len(pair)):
            if pair[ix][1] in bond_num:
                tmp = bond_num[pair[ix][1]]
                tmp.append(pair[ix][2])
                bond_num[pair[ix][1]] = tmp
            else:
                bond_num[pair[ix][1]] = [pair[ix][2]]
            if pair[ix][2] in bond_num:
                tmp = bond_num[pair[ix][2]]
                tmp.append(pair[ix][1])
                bond_num[pair[ix][2]] = tmp
            else:
                bond_num[pair[ix][2]] = [pair[ix][1]]
        for ix in bond_num:
            if len(bond_num[ix]) > 1:
                for jx in range(len(bond_num[ix])):
                    for kx in range(jx + 1, len(bond_num[ix])):
                        if atomtype[bond_num[ix][jx]] + '-' + atomtype[ix] + '-' + atomtype[bond_num[ix][kx]] in \
                                self.angcoeff:
                            ang_p.append([atomtype[bond_num[ix][jx]] + '-' + atomtype[ix] + '-' +
                                          atomtype[bond_num[ix][kx]], bond_num[ix][jx], ix, bond_num[ix][kx]])
                        elif atomtype[bond_num[ix][kx]] + '-' + atomtype[ix] + '-' + atomtype[bond_num[ix][jx]] \
                                in self.angcoeff:
                            ang_p.append([atomtype[bond_num[ix][kx]] + '-' + atomtype[ix] + '-' +
                                          atomtype[bond_num[ix][jx]], bond_num[ix][kx], ix, bond_num[ix][jx]])
                        else:
                            print("no such ang coeff:", atomtype[bond_num[ix][jx]] + '-' +
                                  atomtype[ix] + '-' + atomtype[bond_num[ix][kx]])
        for ix in pair:
            if len(bond_num[ix[1]]) > 1 and len(bond_num[ix[2]]) > 1:
                numn = 0
                for jx in bond_num[ix[1]]:
                    for kx in bond_num[ix[2]]:
                        if jx != ix[2] and kx != ix[1] and jx != kx:
                            if atomtype[jx] + '-' + atomtype[ix[1]] + '-' + atomtype[ix[2]] + '-' + atomtype[kx] in \
                                    self.dihcoeff:
                                dih_p.append([atomtype[jx] + '-' + atomtype[ix[1]] + '-' + atomtype[ix[2]] + '-' +
                                              atomtype[kx], jx, ix[1], ix[2], kx])
                                numn += 1
                            elif atomtype[kx] + '-' + atomtype[ix[2]] + '-' + atomtype[ix[1]] + '-' + atomtype[jx] in \
                                    self.dihcoeff:
                                dih_p.append([atomtype[jx] + '-' + atomtype[ix[1]] + '-' + atomtype[ix[2]] + '-' +
                                              atomtype[kx], jx, ix[1], ix[2], kx])
                                numn += 1
                            else:
                                print("no such dih coeff")
        for ix in pair:
            ix[1] += 1
            ix[2] += 1
        for ix in ang_p:
            ix[1] += 1
            ix[2] += 1
            ix[3] += 1
        for ix in dih_p:
            ix[1] += 1
            ix[2] += 1
            ix[3] += 1
            ix[4] += 1
        self.molcharge.append(charge_p)
        self.molbond.append(pair)
        self.molang.append(ang_p)
        self.moldih.append(dih_p)
        self.molimp.append([])

    def trandata(self, filenm):
        mass_tmp = {}
        pair_tmp = {}
        bond_tmp = {}
        ang_tmp = {}
        dih_tmp = {}
        imp_tmp = {}
        charge_p = []
        bond_p = []
        ang_p = []
        dih_p = []
        imp_p = []
        molxyz = []
        file2 = open("./" + filenm + ".data")
        while 1:
            tt = file2.readline().split()
            if len(tt) > 1 and tt[1] == "atoms":
                break
        whe_bond = int(file2.readline().split()[0])
        whe_ang = int(file2.readline().split()[0])
        whe_dih = int(file2.readline().split()[0])
        whe_imp = int(file2.readline().split()[0])
        while 1:
            tt = file2.readline().split()
            if len(tt) and tt[0] == "Masses":
                break
        file2.readline()
        while 1:
            tt = file2.readline().split()
            if len(tt):
                self.mass[tt[3]] = tt[1]
                mass_tmp[tt[0]] = tt[3]
            else:
                break

        file2.readline()
        file2.readline()
        while 1:
            tt = file2.readline().split()
            if len(tt):
                self.paircoeff[tt[4]] = [tt[1], tt[2]]
                pair_tmp[tt[0]] = tt[4]
            else:
                break
        if whe_bond:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline()
                linesplit = tt.split()
                if len(linesplit):
                    self.bondcoeff[re.split(r'[#\s]', tt)[-2]] = tt.split('#')[0].split()[1:]
                    bond_tmp[linesplit[0]] = re.split(r'[#\s]', tt)[-2]
                else:
                    break
        if whe_ang:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline()
                linesplit = tt.split()
                if len(linesplit):
                    self.angcoeff[re.split(r'[#\s]', tt)[-2]] = tt.split('#')[0].split()[1:]
                    ang_tmp[linesplit[0]] = re.split(r'[#\s]', tt)[-2]
                else:
                    break
        if whe_dih:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline()
                linesplit = tt.split()
                if len(linesplit):
                    self.dihcoeff[re.split(r'[#\s]', tt)[-2]] = tt.split('#')[0].split()[1:]
                    dih_tmp[linesplit[0]] = re.split(r'[#\s]', tt)[-2]
                else:
                    break
        if whe_imp:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline()
                linesplit = tt.split()
                if len(linesplit):
                    self.impcoeff[re.split(r'[#\s]', tt)[-2]] = tt.split('#')[0].split()[1:]
                    imp_tmp[linesplit[0]] = re.split(r'[#\s]', tt)[-2]
                else:
                    break
        file2.readline()
        file2.readline()
        while 1:
            tt = file2.readline().split()
            if len(tt):
                molxyz.append([tt[-1], tt[4], tt[5], tt[6]])
                charge_p.append(float(tt[3]))
            else:
                break
        if whe_bond:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline().split()
                if len(tt):
                    bond_p.append([bond_tmp[tt[1]], int(tt[2]), int(tt[3])])
                else:
                    break
        if whe_ang:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline().split()
                if len(tt):
                    ang_p.append([ang_tmp[tt[1]], int(tt[2]), int(tt[3]), int(tt[4])])
                else:
                    break
        if whe_dih:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline().split()
                if len(tt):
                    dih_p.append([dih_tmp[tt[1]], int(tt[2]), int(tt[3]), int(tt[4]), int(tt[5])])
                else:
                    break
        if whe_imp:
            file2.readline()
            file2.readline()
            while 1:
                tt = file2.readline().split()
                if len(tt):
                    imp_p.append([imp_tmp[tt[1]], int(tt[2]), int(tt[3]), int(tt[4]), int(tt[5])])
                else:
                    break
        file2.close()
        self.molcharge.append(charge_p)
        self.molbond.append(bond_p)
        self.molang.append(ang_p)
        self.moldih.append(dih_p)
        self.molimp.append(imp_p)
        file2 = open(filenm + ".xyz", "w")
        print(len(molxyz), file=file2)
        print(len(molxyz), file=file2)
        for ix in range(len(molxyz)):
            print("%s %.6f %.6f %.6f" % (molxyz[ix][0], float(molxyz[ix][1]),
                                         float(molxyz[ix][2]), float(molxyz[ix][3])), file=file2)
        file2.close()

    def calid(self):
        self.moltot.append(0)
        tmp = 0
        for ix in range(len(self.molnum)):
            tmp += self.molnum[ix] * self.moltimes[ix]
            self.moltot.append(tmp)
        for ix in range(len(self.atomlist)):
            for jx in range(len(self.molnum)):
                if self.moltot[jx+1] > ix >= self.moltot[jx]:
                    molexist = 0
                    for kx in range(jx):
                        molexist += self.moltimes[kx]
                    molid = (ix - self.moltot[jx]) // self.molnum[jx] + 1 + molexist
                    self.atomlist[ix].append(molid)

    def caltype(self):
        for ix in range(len(self.atomlist)):
            if self.atomlist[ix][0] not in self.atomtype:
                self.atomtype.append(self.atomlist[ix][0])
        for ix in range(len(self.bondlist)):
            if self.bondlist[ix][0] not in self.bondtype:
                self.bondtype.append(self.bondlist[ix][0])
        for ix in range(len(self.anglist)):
            if self.anglist[ix][0] not in self.angtype:
                self.angtype.append(self.anglist[ix][0])
        for ix in range(len(self.dihlist)):
            if self.dihlist[ix][0] not in self.dihtype:
                self.dihtype.append(self.dihlist[ix][0])
        for ix in range(len(self.implist)):
            if self.implist[ix][0] not in self.imptype:
                self.imptype.append(self.implist[ix][0])

    def addbond_ang(self):
        for mm in range(0, len(self.molnum)):
            for ix in range(self.moltimes[mm]):
                for jx in range(len(self.molcharge[mm])):
                    self.charge.append(self.molcharge[mm][jx])
                for jx in range(len(self.molbond[mm])):
                    xx = self.molnum[mm]
                    exist = self.moltot[mm]
                    t_1 = self.molbond[mm][jx][0]
                    t_2 = self.molbond[mm][jx][1] + exist + xx * ix
                    t_3 = self.molbond[mm][jx][2] + exist + xx * ix
                    self.bondlist.append([t_1, t_2, t_3])
                for jx in range(len(self.molang[mm])):
                    xx = self.molnum[mm]
                    exist = self.moltot[mm]
                    t_1 = self.molang[mm][jx][0]
                    t_2 = self.molang[mm][jx][1] + exist + xx * ix
                    t_3 = self.molang[mm][jx][2] + exist + xx * ix
                    t_4 = self.molang[mm][jx][3] + exist + xx * ix
                    self.anglist.append([t_1, t_2, t_3, t_4])
                for jx in range(len(self.moldih[mm])):
                    xx = self.molnum[mm]
                    exist = self.moltot[mm]
                    t_1 = self.moldih[mm][jx][0]
                    t_2 = self.moldih[mm][jx][1] + exist + xx * ix
                    t_3 = self.moldih[mm][jx][2] + exist + xx * ix
                    t_4 = self.moldih[mm][jx][3] + exist + xx * ix
                    t_5 = self.moldih[mm][jx][4] + exist + xx * ix
                    self.dihlist.append([t_1, t_2, t_3, t_4, t_5])
                for jx in range(len(self.molimp[mm])):
                    xx = self.molnum[mm]
                    exist = self.moltot[mm]
                    t_1 = self.molimp[mm][jx][0]
                    t_2 = self.molimp[mm][jx][1] + exist + xx * ix
                    t_3 = self.molimp[mm][jx][2] + exist + xx * ix
                    t_4 = self.molimp[mm][jx][3] + exist + xx * ix
                    t_5 = self.molimp[mm][jx][4] + exist + xx * ix
                    self.implist.append([t_1, t_2, t_3, t_4, t_5])

    def output(self, fileout):
        file2 = open(fileout, "w")
        print("the monte", file=file2)
        print(str(len(self.atomlist)) + " atoms", file=file2)
        print(str(len(self.bondlist)) + " bonds", file=file2)
        print(str(len(self.anglist)) + " angles", file=file2)
        print(str(len(self.dihlist)) + " dihedrals", file=file2)
        print(str(len(self.implist)) + " impropers", file=file2)
        print(str(len(self.atomtype)) + " atom types", file=file2)
        print(str(len(self.bondtype)) + " bond types", file=file2)
        print(str(len(self.angtype)) + " angle types", file=file2)
        print(str(len(self.dihtype)) + " dihedral types", file=file2)
        print(str(len(self.imptype)) + " improper types\n", file=file2)
        print("%.5f %.5f %s" % (0, self.box[0], " xlo xhi"), file=file2)
        print("%.5f %.5f %s" % (0, self.box[1], " ylo yhi"), file=file2)
        print("%.5f %.5f %s" % (0, self.box[2] + 3, " zlo zhi"), file=file2)
        print("\nMasses\n", file=file2)
        for i in range(len(self.atomtype)):
            print("%d %.6f %s" % (i + 1, float(self.mass[self.atomtype[i]]), "# " + self.atomtype[i]), file=file2)
        print("\nPair Coeffs\n", file=file2)
        for i in range(len(self.atomtype)):
            print("%d %.10f %.10f %s" % (i + 1, float(self.paircoeff[self.atomtype[i]][0]),
                                         float(self.paircoeff[self.atomtype[i]][1]), "# "+self.atomtype[i]),
                  file=file2)
        print("\nBond Coeffs\n", file=file2)
        for i in range(len(self.bondtype)):
            print("%d %s %s" % (i + 1, ' '.join(self.bondcoeff[self.bondtype[i]]), "# " + self.bondtype[i]),
                  file=file2)
        len_ang_type = len(set([self.angcoeff[self.angtype[i]][0] for i in range(len(self.angtype))]))
        if len_ang_type > 1:
            print("\nAngle Coeffs # hybrid\n", file=file2)
            for i in range(len(self.angtype)):
                print("%d %s %s" % (i + 1, ' '.join(self.angcoeff[self.angtype[i]]), "# " + self.angtype[i]),
                      file=file2)
        else:
            print("\nAngle Coeffs # %s \n" % self.angcoeff[self.angtype[0]][0], file=file2)
            for i in range(len(self.angtype)):
                print("%d %s %s" % (i + 1, ' '.join(self.angcoeff[self.angtype[i]][1:]), "# " + self.angtype[i]),
                      file=file2)
        if len(self.dihtype) > 0:
            for i in range(len(self.dihtype)):
                if self.dihcoeff[self.dihtype[i]][0] == "charmm":
                    self.dihcoeff[self.dihtype[i]][3] = str(int(float(self.dihcoeff[self.dihtype[i]][3])))
            len_dih_type = len(set([self.dihcoeff[self.dihtype[i]][0] for i in range(len(self.dihtype))]))
            if len_dih_type > 1:
                print("\nDihedral Coeffs # hybrid\n", file=file2)
                for i in range(len(self.dihtype)):
                    print("%d %s %s" % (i + 1, ' '.join(self.dihcoeff[self.dihtype[i]]), "# " + self.dihtype[i]),
                          file=file2)
            else:
                print("\nDihedral Coeffs # %s \n" % self.dihcoeff[self.dihtype[0]][0], file=file2)
                for i in range(len(self.dihtype)):
                    print("%d %s %s" % (i + 1, ' '.join(self.dihcoeff[self.dihtype[i]][1:]), "# " + self.dihtype[i]),
                          file=file2)
        if len(self.imptype) > 0:
            print("\nImproper Coeffs\n", file=file2)
            for i in range(len(self.imptype)):
                print("%d %s %s" % (i + 1, ' '.join(self.impcoeff[self.imptype[i]]), "# " + self.imptype[i]),
                      file=file2)
        print("\nAtoms\n", file=file2)
        for i in range(len(self.atomlist)):
            print("%d %d  %d %.6f %.6f %.6f %.6f " % (
                i + 1, self.atomlist[i][4], self.atomtype.index(self.atomlist[i][0]) + 1,
                float(self.charge[i]), float(self.atomlist[i][1]), float(self.atomlist[i][2]),
                float(self.atomlist[i][3])), "# ", self.atomlist[i][0], file=file2)
        print("\nBonds\n", file=file2)
        for i in range(len(self.bondlist)):
            print("%d %d  %d %d" % (i + 1, self.bondtype.index(self.bondlist[i][0]) + 1,
                                    self.bondlist[i][1], self.bondlist[i][2]), file=file2)
        print("\nAngles\n", file=file2)
        for i in range(len(self.anglist)):
            print("%d %d  %d %d %d" % (i + 1, self.angtype.index(self.anglist[i][0]) + 1,
                                       self.anglist[i][1], self.anglist[i][2], self.anglist[i][3]), file=file2)
        if len(self.dihtype) > 0:
            print("\nDihedrals\n", file=file2)
            for i in range(len(self.dihlist)):
                print("%d %d  %d %d %d %d" % (i + 1, self.dihtype.index(self.dihlist[i][0]) + 1,
                                              self.dihlist[i][1], self.dihlist[i][2], self.dihlist[i][3],
                                              self.dihlist[i][4]), file=file2)
        if len(self.imptype) > 0:
            print("\nImpropers\n", file=file2)
            for i in range(len(self.implist)):
                print("%d %d  %d %d %d %d" % (i + 1, self.imptype.index(self.implist[i][0]) + 1,
                                              self.implist[i][1], self.implist[i][2], self.implist[i][3],
                                              self.implist[i][4]), file=file2)
        file2.close()


FinalLmpData("./"+"pyro_xyz", add_z=[[['li', 23, 5.0, 24], ["h2o", 18, 1.1, 300]],
                                     [['li', 23, 5.0, 24], ["h2o", 18, 1.1, 300]],
                                     [['li', 23, 5.0, 24], ["h2o", 18, 1.1, 300]],
                                     [['li', 23, 5.0, 24], ["h2o", 18, 1.1, 300]]], add_y=[])
