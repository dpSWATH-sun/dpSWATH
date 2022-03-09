# -*- coding: utf-8 -*-

import os
import re

import datetime

from multiprocessing import Pool


import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET

from scipy.stats import pearsonr
from scipy.stats import spearmanr

import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer

from keras import backend as K
from keras.models import Input, Model
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Bidirectional, LSTM, Activation, Dropout, MaxPooling1D
from keras.layers import Dense, TimeDistributed, concatenate, Masking,merge
from keras.optimizers import SGD
import tensorflow as tf
from keras.layers.core import Flatten

from css321 import *
from seq_self_attention import SeqSelfAttention

from keras.utils import CustomObjectScope
from keras.models import load_model


import pickle
from scipy import sparse

from tqdm import tqdm

##############################################
from math import sqrt
def RMSE(act, pred):
    return sqrt(np.mean(np.square(act - pred)))

from scipy.stats import pearsonr
def Pearson(act, pred):
    return pearsonr(act, pred)[0]

from scipy.stats import spearmanr
def Spearman(act, pred):
    return spearmanr(act, pred)[0]

def Delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]

def Delta_tr95(act, pred):
    return Delta_t95(act, pred) / (max(act) - min(act))

###############################################

def AAVecDict():
    aa_vec = {}
    s = "0ACDEFGHIKLMNPQRSTVWY"
    v = [0] * len(s)
    v[0] = 1
    for i in range(len(s)):
        aa_vec[s[i]] = list(v)
        v[i], v[(i + 1) % 21] = 0, 1
    return aa_vec

def ChgVecDict():
    chg_vec = {}
    d = "0123456"
    p = [0] * len(d)
    p[0] = 1
    for i in range(len(d)):
        chg_vec[d[i]] = list(p)
        p[i], p[(i + 1) % 7] = 0, 1
    return chg_vec

def OnehotEncod_ms(seq, chg):
    one_hot_encod_ms = np.zeros((60, 28), dtype=np.float32)
    for i in range(0, len(seq)):
        one_hot_encod_ms[i, 0:21] = AAVecDict()[seq[i]]
        one_hot_encod_ms[i, 21:28] = ChgVecDict()[str(chg)]
    return one_hot_encod_ms

def OnehotEncod_rt(seq):
    one_hot_encod_rt = np.zeros((60, 21), dtype=np.float32)
    for i in range(0, len(seq)):
        one_hot_encod_rt[i, 0:21] = AAVecDict()[seq[i]]
    return one_hot_encod_rt

def paddingZero(itn_vec):
    for j in range(60 - len(itn_vec) - 1):
        itn_vec = np.concatenate((itn_vec, np.array([[0] * 2 * 2], dtype=np.float32)), axis=0)
    return (itn_vec)

def Mass(seq, chrg):
    AA_vec = {}
    AA_vec['A'] = 71.03711
    AA_vec['R'] = 156.10111
    AA_vec['N'] = 114.04293
    AA_vec['D'] = 115.02694
    AA_vec['C'] = 160.03069
    AA_vec['E'] = 129.04259
    AA_vec['Q'] = 128.05858
    AA_vec['G'] = 57.02146
    AA_vec['H'] = 137.05891
    AA_vec['I'] = 113.08406
    AA_vec['L'] = 113.08406
    AA_vec['K'] = 128.09496
    AA_vec['M'] = 131.04049
    AA_vec['F'] = 147.06841
    AA_vec['P'] = 97.05276
    AA_vec['S'] = 87.03203
    AA_vec['T'] = 101.04768
    AA_vec['W'] = 186.07931
    AA_vec['Y'] = 163.06333
    AA_vec['V'] = 99.06841

    Mb = 1.0078
    My = 19.0184
    mss = []
    pr_mss = (np.sum([AA_vec[i] for i in seq]) + 18.0106 + 1.0078 * chrg) / chrg
    for i in range(1, len(seq)):
        for iname in ['y', 'b']:
            for chrg in range(1, 2 + 1):
                if iname == 'y':
                    if chrg == 1:
                        mss.append((np.sum([AA_vec[i] for i in seq[-i:]]) + My) / chrg)

                    else:
                        mss.append((np.sum([AA_vec[i] for i in seq[-i:]]) + My + Mb*(chrg-1)) / chrg)
                elif iname == 'b':
                    if chrg == 1:
                        mss.append((np.sum([AA_vec[i] for i in seq[:i]]) + Mb) / chrg)
                    else:
                        mss.append((np.sum([AA_vec[i] for i in seq[:i]]) + Mb + Mb*(chrg-1)) / chrg)

    return mss,pr_mss


def dpRT_train(bld_md_rt_one,ions_rt10one,nrts10one,md_rt_dir_one):
    md_rt = load_model(bld_md_rt_one, custom_objects=SeqSelfAttention.get_custom_objects())

    rt_epohs = 30
    rt_batches = 128

    md_rt_h5_one = md_rt_dir_one + 'dpRT_'+'_%03d_%.5f_%.5f.h5'

    for j in range(1,rt_epohs+1):
        hsty = md_rt.fit(ions_rt10one, nrts10one, validation_split=0.05, epochs=1, batch_size=rt_batches, verbose=1)
        mdd_rt_one = md_rt_h5_one % (j, hsty.history['loss'][0],hsty.history['val_loss'][0])
        md_rt.save(mdd_rt_one)


def dpMS_train(bld_md_ms_one,ions20one,itens20one,md_ms_dir_one):
    md_ms_one = load_model(bld_md_ms_one, custom_objects=SeqSelfAttention.get_custom_objects())

    ms_epohs = 30
    ms_batches = 128

    md_ms_h5_one = md_ms_dir_one + 'dpMS_%03d_%.5f_%.5f.h5'
    for j in range(1,ms_epohs+1):
        hsty = md_ms_one.fit(ions20one, itens20one, validation_split=0.05, epochs=1, batch_size=ms_batches, verbose=1)
        mdd_ms_one = md_ms_h5_one % (j, hsty.history['loss'][0],hsty.history['val_loss'][0])
        md_ms_one.save(mdd_ms_one)


def wrtf_one(x):
    lines, file_path = x

    with open(file_path, 'a+') as f:
        f.writelines(lines)

def sele_ct21one(i, df0, cnt2, file_path, n):
    lines = []
    for x in range(i * n, (i + 1) * n):
        dfl = df0[df0.iloc[:, 8].isin([cnt2.index[x]])]
        for ct in np.array(dfl):
            line = '\t'.join(map(str, list(ct[0:4] + [ct[5]]))) + '\t' + ct[8] + '\n'
            lines.append(line)

    return (lines, file_path)

def sele_ct22one(i, df0, cnt2, file_path, y, n):
    lines = []
    for x in range(i * n, i * n + y):
        dfl = df0[df0.iloc[:, 8].isin([cnt2.index[x]])]
        for ct in np.array(dfl):
            line = '\t'.join(map(str, list(ct[0:4] + [ct[5]]))) + '\t' + ct[8] + '\n'
            lines.append(line)

    return (lines, file_path)

def sele_ct31one(i, df0, cnt3, file_path, n):
    lines = []
    for x in range(i * n, (i + 1) * n):
        dfl = df0[df0.iloc[:, 7].isin([cnt3.index[x]])]

        seq = cnt3.index[x].split('_')[0]
        charge = cnt3.index[x].split('_')[1]
        seq_chg = cnt3.index[x]
        seql = len(seq)
        et2 = []

        for j in range(cnt3[x]):

            ty_itn = {}
            ak = dfl.iloc[j, 2].split(';')
            bk = [float(i) for i in dfl.iloc[j, 3].split(';')]

            for k in range(len(ak)):
                if "^" in ak[k] and int(ak[k][ak[k].find("^") + 1:]) > 2: continue
                if str(seql) in ak[k]: continue
                if "^" in ak[k] and int(ak[k][ak[k].find("^") + 1:]) >= int(charge): continue

                ty_itn[ak[k]] = bk[k]

            if len(ty_itn) < seql / 2: continue
            # if len(ty_itn) < seql: continue

            bar0 = np.max(list(ty_itn.values()))

            ak = list(ty_itn.keys())
            bk = [v / bar0 for v in list(ty_itn.values())]

            et2.extend(ty_itn.keys())
            pep1_dic = dict(zip(ak, bk))

            if j == 0:
                pep_df = pd.DataFrame(pep1_dic, index=[0])
            else:
                pep1_df = pd.DataFrame(pep1_dic, index=[j])
                pep_df = pd.concat([pep_df, pep1_df], ignore_index=True)

            pep_df = pep_df.fillna(0)

        etc = pd.value_counts(et2)
        etc_filter = etc[etc > cnt3[x] * 0.5]
        pep_dff = pd.DataFrame(pep_df, columns=etc_filter.index)

        rsts = list()
        for p in np.arange(0.01, 0.21, 0.01):
            rst = Cs321(p, pep_dff).hclust()
            if rst[4] == 1:
                rsts.append(rst)
                break
            else:
                rsts.append(rst)

        rsts_df = pd.DataFrame(rsts)

        fp = list(map(int, rsts_df[rsts_df.iloc[:, 3] == np.max(rsts_df.iloc[:, 3])].iloc[0, 1].split(';')))

        # if len(fp) == 1:
        #     continue
        # else:
        #     pep_fdf = pep_dff.iloc[fp,:]

        pep_fdf = pep_dff.iloc[fp, :]

        ions_names = ';'.join(etc_filter.index)
        for lc in range(pep_fdf.shape[0]):
            lci = ';'.join(list(map(str, pep_fdf.iloc[lc, :])))
            line = seq + '\t' + charge + '\t' + ions_names + '\t' + lci + '\t' + seq_chg + '\n'
            lines.append(line)

    return (lines, file_path)

def sele_ct32one(i, df0, cnt3, file_path, y, n):
    lines = []
    for x in range(i * n, i * n + y):
        dfl = df0[df0.iloc[:, 7].isin([cnt3.index[x]])]
        seq = cnt3.index[x].split('_')[0]
        charge = cnt3.index[x].split('_')[1]
        seq_chg = cnt3.index[x]
        seql = len(seq)
        et2 = []

        for j in range(cnt3[x]):
            ty_itn = {}
            ak = dfl.iloc[j, 2].split(';')
            bk = [float(i) for i in dfl.iloc[j, 3].split(';')]

            for k in range(len(ak)):
                if "^" in ak[k] and int(ak[k][ak[k].find("^") + 1:]) > 2: continue
                if str(seql) in ak[k]: continue
                if "^" in ak[k] and int(ak[k][ak[k].find("^") + 1:]) >= int(charge): continue

                ty_itn[ak[k]] = bk[k]

            if len(ty_itn) < seql / 2: continue
            # if len(ty_itn) < seql: continue

            bar0 = np.max(list(ty_itn.values()))

            ak = list(ty_itn.keys())
            bk = [v / bar0 for v in list(ty_itn.values())]

            et2.extend(ty_itn.keys())
            pep1_dic = dict(zip(ak, bk))

            if j == 0:
                pep_df = pd.DataFrame(pep1_dic, index=[0])
            else:
                pep1_df = pd.DataFrame(pep1_dic, index=[j])
                pep_df = pd.concat([pep_df, pep1_df], ignore_index=True)

            pep_df = pep_df.fillna(0)

        etc = pd.value_counts(et2)
        etc_filter = etc[etc > cnt3[x] * 0.5]
        pep_dff = pd.DataFrame(pep_df, columns=etc_filter.index)

        rsts = list()
        for p in np.arange(0.01, 0.21, 0.01):
            rst = Cs321(p, pep_dff).hclust()
            if rst[4] == 1:
                rsts.append(rst)
                break
            else:
                rsts.append(rst)

        rsts_df = pd.DataFrame(rsts)

        fp = list(map(int, rsts_df[rsts_df.iloc[:, 3] == np.max(rsts_df.iloc[:, 3])].iloc[0, 1].split(';')))

        # if len(fp) == 1:
        #     continue
        # else:
        #     pep_fdf = pep_dff.iloc[fp,:]

        pep_fdf = pep_dff.iloc[fp, :]

        ions_names = ';'.join(etc_filter.index)
        for lc in range(pep_fdf.shape[0]):
            lci = ';'.join(list(map(str, pep_fdf.iloc[lc, :])))
            line = seq + '\t' + charge + '\t' + ions_names + '\t' + lci + '\t' + seq_chg + '\n'
            lines.append(line)

    return (lines, file_path)

def bld(bld_dat_pep_one,md_rt_one,md_ms_one):
    ions_ms3one = []
    ions_rt3one = []
    seq_chg3one = []

    ions_ms36one = []
    ions_rt36one = []
    seq_chg36one = []

    ffile7one = []

    print('Reading precursors...')
    ffileo7one = open(''.join(bld_dat_pep_one)).readlines()
    ffile7one.extend(ffileo7one)

    print('Formatting precursors...')
    for dpl in ffile7one:
        if dpl == "": break

        dpi = dpl.split("\t")
        seq = dpi[0]
        chg = int(dpi[1])

        if chg > 6: continue
        if len(seq) > 60: continue
        if len(seq) < 7: continue

        ion_ms = OnehotEncod_ms(seq, chg)
        ion_rt = OnehotEncod_rt(seq)

        ions_ms3one.append(ion_ms)
        ions_rt3one.append(ion_rt)
        seq_chg36one.append(dpl.strip('\n'))

    ions_ms36one.extend(ions_ms3one)
    ions_rt36one.extend(ions_rt3one)
    seq_chg36one.extend(seq_chg3one)

    ions_ms36one = np.array(ions_ms36one)
    ions_rt36one = np.array(ions_rt36one)

    print(ions_ms36one.shape)
    print(ions_rt36one.shape)

    print('Formatting precursors done.')
    print('Predicting retention time...')

    mdi_pred_rt_one1 = load_model(md_rt_one, custom_objects=SeqSelfAttention.get_custom_objects())
    rt_pred_lib_one1 = mdi_pred_rt_one1.predict(ions_rt36one)

    rt_mins_one1 = float(md_rt_one.split('/')[-1].split('_')[1])
    rt_maxs_one1 = float(md_rt_one.split('/')[-1].split('_')[2])
    print(rt_mins_one1, rt_maxs_one1)

    rt_pred_lib_one1 = rt_pred_lib_one1 * (rt_maxs_one1 - rt_mins_one1) + rt_mins_one1

    rt_pred_lib_one1 = rt_pred_lib_one1.flatten()

    print('Predicting intensities...')

    mdi_pred_ms_one1 = load_model(md_ms_one, custom_objects=SeqSelfAttention.get_custom_objects())
    ms_pred_lib_one1 = mdi_pred_ms_one1.predict(ions_ms36one)

    libDir_one1 = os.getcwd() + '/dpSWATH/Library/'
    libPath_one1 = os.getcwd() + '/dpSWATH/Library/dpSWATH-Lib.txt'

    if os.path.exists(libDir_one1):
        pass
    else:
        os.makedirs(libDir_one1)

    fs = open(libPath_one1, "a+")

    fs.writelines('StrippedSequence\tModifiedSequence\tPrecursorCharge\tiRT\tPrecursorMz\tFragmentMz\tFragmentType\tFragmentNumber\tFragmentCharge\tRelativeFragmentIntensity\n')

    count8 = 0
    print('Writing transitions...')
    # rows36one = len(seq_chg36one)
    for dpl in seq_chg36one:

        dpi = dpl.split("\t")
        seq = dpi[0]
        modseq = seq
        chg = int(dpi[1])

        if 'C' in seq:
            modseq = modseq.replace('C', 'C[Carbamidomethyl (C)]')

        count8 += 1
        itnm = []

        for i in range(1, len(seq)):
            for iname in ['y', 'b']:
                for chrg in range(1, 2 + 1):
                    if chrg == 1:
                        ioname = "{}{}^{}".format(iname, i, 1)
                        itnm.append(ioname)
                    else:
                        ioname = "{}{}^{}".format(iname, i, chrg)
                        itnm.append(ioname)

        mssl, pr_mss = Mass(seq, chg)

        if chg == 2:
            mssm = np.array(mssl, dtype=np.float32).reshape((len(seq) - 1), 2 * 2)[
                   :(len(seq) - 1), (0, 2)].reshape(1, 2 * (len(seq) - 1), order='C')
            itnms = np.array(itnm).reshape((len(seq) - 1), 2 * 2)[:(len(seq) - 1),
                    (0, 2)].reshape(1, 2 * (len(seq) - 1), order='C')
            itns = np.array(ms_pred_lib_one1[count8 - 1], dtype=np.float32).reshape((60 - 1),
                                                                               2 * 2)[
                   :(len(seq) - 1), (0, 2)].reshape(1, 2 * (len(seq) - 1), order='C')

        else:
            mssm = np.array(mssl, dtype=np.float32).reshape((len(seq) - 1), 2 * 2)[
                   :(len(seq) - 1), (0, 1, 2, 3)].reshape(1, 4 * (len(seq) - 1), order='C')
            itnms = np.array(itnm).reshape((len(seq) - 1), 2 * 2)[:(len(seq) - 1),
                    (0, 1, 2, 3)].reshape(1, 4 * (len(seq) - 1), order='C')
            itns = np.array(ms_pred_lib_one1[count8 - 1], dtype=np.float32).reshape((60 - 1),
                                                                               2 * 2)[
                   :(len(seq) - 1), (0, 1, 2, 3)].reshape(1, 4 * (len(seq) - 1), order='C')

        lines = []
        for j in range(len(mssm[0])):
            if itns[0, j] == 0: continue
            line = seq + '\t' + modseq + '\t' + str(chg) + '\t' + str(rt_pred_lib_one1[count8 - 1]) + '\t' + str(
                pr_mss) + '\t' + str(mssm[0, j]) + '\t' + itnms[0, j].split('^')[0][0] + '\t' + str(
                itnms[0, j].split('^')[0][1:]) + '\t' + str(itnms[0, j].split('^')[1]) + '\t' + str(
                format(itns[0, j], '0.10f')) + '\n'

            lines.append(line)

        lines_sorted = sorted(lines, key=lambda line: line.split('\t')[9], reverse=True)

        if lines_sorted:
            if float(lines_sorted[0].split('\t')[9].split('\n')[0]):
                if len(lines_sorted) >= 6:
                    lines_top6 = lines_sorted[0:6]
                    fs.writelines(lines_top6)
                else:
                    fs.writelines(lines_sorted)
            else:
                pass

    fs.close()

def main():

    while True:
        options = input('Please select the number of functions in dpSWATH：1.train; 2.build library; 3.exit\n')

        if options == '1':
            set_wd = input('Please set your working directory:')
            if os.path.exists(set_wd):
                os.chdir(set_wd)
            else:
                print('Please check your directory and try agian!')
                break

            options1_1 = input('Which model do you want to train?: 1.dpRT; 2.dpMS; 3.exit\n')

            if options1_1 == '1':
                bld_md_rt_one = input('Please select the pretrained model for fine-tuning[.h5]:')
                dats_rt_sp = input('Please select the DDA library built by Spectronaut[.xls]:')

                print('Loading data for training of dpRT...')

                rt_df = pd.read_table(dats_rt_sp,encoding= 'unicode_escape')
                rt_dp = rt_df[['StrippedPeptide', 'iRT']].drop_duplicates()
                rt_min_one = np.min(rt_dp.iloc[:,1]) - 10
                rt_max_one = np.max(rt_dp.iloc[:,1]) + 10
                ions_rt = []
                nrts = []

                print('Formatting peptides for dpRT...')

                rows = rt_dp.shape[0]
                for ip in range(0, rows):
                    dpl = rt_dp.iloc[ip, :]

                    if dpl.empty: break

                    seq = dpl.iloc[0]
                    if len(seq) <= 60:

                        rt = dpl.iloc[1]

                        nrt = (float(rt) - rt_min_one) / (rt_max_one - rt_min_one)

                        ion_rt = OnehotEncod_rt(seq)
                        ions_rt.append(ion_rt)
                        nrts.append(nrt)
                    else:
                        pass

                ions_rt10one = np.array(ions_rt)
                nrts10one = np.array(nrts)

                print('Formatting peptides for dpRT done.')
                print('Training dpRT model...')

                time_now = datetime.datetime.now()
                md_rt_dir = os.getcwd() + '/dpSWATH/md/dpRT/' + re.sub(':|\\.', '_', '_'.join(str(time_now).split(' '))) + '/'
                if os.path.exists(md_rt_dir):
                    pass
                else:
                    os.makedirs(md_rt_dir)

                dpRT_train(bld_md_rt_one,ions_rt10one,nrts10one,md_rt_dir)

                print('dpRT training done.')

            elif options1_1 == '2':
                bld_md_ms_one = input('Please select the pretrained model for fine-tuning[.h5]:')
                dats_ms_pp = input('Please select the mass spectra file generated from ProteinPilot[.mzid]:')

                print('Loading data for training of dpMS...')

                tree = ET.ElementTree(file=dats_ms_pp)
                root = tree.getroot()

                ms_dir = os.getcwd() + '/dpSWATH/data/MS_tmp'
                if os.path.exists(ms_dir):
                    pass
                else:
                    os.makedirs(ms_dir)
                mzid_file_path = os.getcwd() + '/dpSWATH/data/MS_tmp/mzid_MS.dp'

                with open(mzid_file_path, 'a+') as f:
                    for i in range(1, len(root[8][1][0])):
                        for ii in range(0, len(root[8][1][0][i]) - 3):
                            seq = root[8][1][0][i][ii].attrib['peptide_ref']

                            p = re.compile(r'\[.{3}\]')
                            seql = p.findall(seq)

                            if '[CAM]' in seql and len(np.unique(seql)) == 1 and '-' not in seq or len(seql) == 0:

                                pp = re.compile(r'(?<!C)\[CAM\]')
                                seqll = pp.findall(seq)

                                ppp = re.compile(r'(?<!\[)C(?!\[CAM\])')
                                seqlll = ppp.findall(seq)
                                if len(seqll) == 0 and len(seqlll) == 0 and not bool(re.match('X|U', seq)):
                                    seq = seq.replace('[CAM]', '')
                                    charge = int(root[8][1][0][i][ii].attrib['chargeState'])
                                    if len(seq) >= 7 and len(seq) <= 60 and charge >= 2 and charge <= 6:
                                        ions = str()
                                        mzs = str()
                                        itns = str()

                                        ll = len(root[8][1][0][i][ii])
                                        conf = float(root[8][1][0][i][ii][ll - 2].attrib['value'])
                                        for j in range(0, len(root[8][1][0][i][ii][ll - 3])):
                                            mzs00 = root[8][1][0][i][ii][ll - 3][j][0].attrib['values']
                                            mzs0 = ';'.join([s for s in mzs00.replace(' ', ';').split(';')]) + ';'
                                            itns00 = root[8][1][0][i][ii][ll - 3][j][1].attrib['values']
                                            itns0 = ';'.join([s for s in itns00.replace(' ', ';').split(';')]) + ';'

                                            if root[8][1][0][i][ii][ll - 3][j].attrib['charge'] == '1':
                                                ions_name = \
                                                root[8][1][0][i][ii][ll - 3][j][2].attrib['name'].split(' ')[1]
                                                ions_idx = root[8][1][0][i][ii][ll - 3][j].attrib['index']
                                                ions0 = ';'.join(
                                                    [ions_name + s for s in
                                                     ions_idx.replace(' ', ';').split(';')]) + ';'
                                            else:
                                                ions_chg = root[8][1][0][i][ii][ll - 3][j].attrib['charge']
                                                ions_name = \
                                                root[8][1][0][i][ii][ll - 3][j][2].attrib['name'].split(' ')[1]
                                                ions_idx = root[8][1][0][i][ii][ll - 3][j].attrib['index']
                                                ions0 = ';'.join([ions_name + s + '^' + ions_chg for s in
                                                                  ions_idx.replace(' ', ';').split(';')]) + ';'

                                            ions = ions + ions0
                                            mzs = mzs + mzs0
                                            itns = itns + itns0

                                        ions = ions.rstrip(';')
                                        mzs = mzs.rstrip(';')
                                        itns = itns.rstrip(';')

                                        line = seq + '\t' + str(
                                            charge) + '\t' + ions + '\t' + itns + '\t' + mzs + '\t' + str(conf) + '\n'

                                        f.writelines(line)
                                else:
                                    pass

                print('Loading MS information...')

                df = pd.read_table(''.join(mzid_file_path), sep='\t', header=None,encoding= 'unicode_escape')

                len_seq = [len(df.iloc[i, 0]) for i in range(df.shape[0])]
                len_ions = [len(df.iloc[i, 2].split(';')) for i in range(df.shape[0])]
                df['6'] = len_seq
                df['7'] = len_ions

                df0 = df[(df.iloc[:, 7] >= df.iloc[:, 6])]
                seq_charge = [str(df0.iloc[i, 0] + '_' + str(df0.iloc[i, 1])) for i in range(df0.shape[0])]

                df0['8'] = seq_charge
                cnt = df0['8'].value_counts()

                cnt3 = cnt[cnt >= 3]
                cnt2 = cnt[cnt < 3]

                dfnf_path_one = os.getcwd() + '/dpSWATH/data/psd/MS/MS_psd.dp'
                dfnf_dir_one = os.getcwd() + '/dpSWATH/data/psd/MS'

                if os.path.exists(dfnf_dir_one):
                    pass
                else:
                    os.makedirs(dfnf_dir_one)


                prs_ms_cores_one = 10

                n2 = len(cnt2) // prs_ms_cores_one
                x2 = prs_ms_cores_one
                y2 = len(cnt2) - x2 * n2

                n3 = len(cnt3) // prs_ms_cores_one
                x3 = prs_ms_cores_one
                y3 = len(cnt3) - x3 * n3

                print('Calculating dpMScore...')

                pool1one = Pool(prs_ms_cores_one)
                for i in range(x2):
                    pool1one.apply_async(sele_ct21one, (i, df0, cnt2, dfnf_path_one, n2), callback=wrtf_one)
                pool1one.apply_async(sele_ct22one, (x2, df0, cnt2, dfnf_path_one, y2, n2), callback=wrtf_one)

                for j in range(x3):
                    pool1one.apply_async(sele_ct31one, (j, df0, cnt3, dfnf_path_one, n3), callback=wrtf_one)
                pool1one.apply_async(sele_ct32one, (x3, df0, cnt3, dfnf_path_one, y3, n3), callback=wrtf_one)

                pool1one.close()
                pool1one.join()


                ions2one = []
                itens2one = []

                ions20one = []
                itens20one = []

                ffile2one = []
                print('Loading MS parsed file...')

                ffileo2one = open(dfnf_path_one).readlines()
                ffile2one.extend(ffileo2one)

                print('Formatting peptides for dpMS...')
                for dpl in ffile2one:
                    if dpl == "": break

                    ty_itn = {}

                    dpi = dpl.split("\t")
                    seq = dpi[0]
                    chg = int(dpi[1])
                    ion_ty = dpi[2].split(";")
                    itns = [float(i) for i in dpi[3].split(";")]

                    if chg > 6: continue
                    if len(seq) > 60: continue
                    if len(seq) < 7: continue
                    if len(itns) < len(seq): continue

                    for i in range(len(ion_ty)):
                        if "^" in ion_ty[i] and int(ion_ty[i][ion_ty[i].find("^") + 1:]) > 2: continue
                        if str(len(seq)) in ion_ty[i]: continue
                        if "^" in ion_ty[i] and int(ion_ty[i][ion_ty[i].find("^") + 1:]) >= chg: continue

                        ty_itn[ion_ty[i]] = itns[i]

                    if len(ty_itn) < len(seq) : continue
                    bar0 = max(list(ty_itn.values()))
                    ion = OnehotEncod_ms(seq, chg)

                    itn = []

                    for i in range(1, len(seq)):
                        for iname in ['y', 'b']:
                            for chrg in range(1, 2 + 1):
                                if chrg == 1:
                                    ioname = "{}{}".format(iname, i)
                                    if ioname in ty_itn:
                                        itn.append(ty_itn[ioname] / bar0)
                                    else:
                                        itn.append(0)
                                else:
                                    ioname = "{}{}^{}".format(iname, i, chrg)
                                    if ioname in ty_itn:
                                        itn.append(ty_itn[ioname] / bar0)
                                    else:
                                        itn.append(0)

                    itns = np.array(itn, dtype=np.float32).reshape(len(seq) - 1, 2 * 2)
                    itns = paddingZero(itns)

                    ions2one.append(ion)
                    itens2one.append(itns)

                ions20one.extend(ions2one)
                itens20one.extend(itens2one)

                ions20one = np.array(ions20one)
                itens20one = np.array(itens20one)

                print('Formatting peptides for dpMS done.')
                print('Training dpMS model...')

                time_now = datetime.datetime.now()
                md_ms_dir = os.getcwd() + '/dpSWATH/md/dpMS/' + re.sub(':|\\.', '_', '_'.join(str(time_now).split(' '))) + '/'
                if os.path.exists(md_ms_dir):
                    pass
                else:
                    os.makedirs(md_ms_dir)

                dpMS_train(bld_md_ms_one,ions20one,itens20one,md_ms_dir)

                print('dpMS training done.')

            elif options1_1 == '3':
                print('Thanks for using dpSWATH!')
                break

        elif options == '2':
            set_wd = input('Please set your working directory:')
            if os.path.exists(set_wd):
                os.chdir(set_wd)
            else:
                print('Please check your directory and try agian!')

            lib_dp_dir = os.getcwd() + '/dpSWATH/Library/'
            if os.path.exists(lib_dp_dir):
                pass
            else:
                os.makedirs(lib_dp_dir)

            lists_pred = input('Please select the file of precursors for building library[.dp/.txt]:')
            md_rt_pred = input('Please select the trained dpRT model for building library[.h5]:')
            md_ms_pred = input('Please select the trained dpMS model for building library[.h5]:')

            bld(lists_pred,md_rt_pred,md_ms_pred)
            print('dpSWATH library building done.')

        elif options == '3':
            print('Thanks for using dpSWATH！')
            break
        else:
            print('Invalid input, please try again!')

if __name__ == '__main__':
    main()












