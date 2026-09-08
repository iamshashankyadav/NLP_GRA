BROWN_TO_PTB = {
    'AT':'DT','DT':'DT','DTI':'DT','DTS':'DT','DTX':'DT','ABN':'DT','ABX':'DT',
    'NN':'NN','NNS':'NNS','NP':'NNP','NPS':'NNPS','NR':'NN','NN$':'NN','NP$':'NNP','NNS$':'NNS',
    'JJ':'JJ','JJR':'JJR','JJT':'JJS','AP':'JJ','OD':'JJ',
    'VB':'VB','VBD':'VBD','VBG':'VBG','VBN':'VBN','VBZ':'VBZ','VBP':'VBP',
    'BE':'VB','BED':'VBD','BEDZ':'VBD','BEG':'VBG','BEM':'VBP','BEN':'VBN','BER':'VBP','BEZ':'VBZ',
    'HV':'VB','HVD':'VBD','HVG':'VBG','HVN':'VBN','HVZ':'VBZ',
    'DO':'VB','DOD':'VBD','DOZ':'VBZ',
    'IN':'IN','CS':'IN','CC':'CC','CD':'CD','TO':'TO','MD':'MD','RP':'RP','UH':'UH',
    'PN':'PRP','PPO':'PRP','PPL':'PRP','PPLS':'PRP','PPSS':'PRP','PPS':'PRP','PP$':'PRP$',
    'RB':'RB','RBR':'RBR','RBT':'RBS','QL':'RB','ABL':'RBR',
    'WDT':'WDT','WPS':'WP','WPO':'WP','WP$':'WP$','WRB':'WRB','EX':'EX','FW':'FW'
}

def map_brown_to_ptb(tag):
    base = tag.split('-')[0].split('+')[0]
    return BROWN_TO_PTB.get(base, 'NN')
