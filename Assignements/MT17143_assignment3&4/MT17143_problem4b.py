#PROFILE BRANCHING ALGORITHM
#MT17143
#AKSHITA SAWHNEY

import random
import math

def profile_branching(set_of_sequences,size_of_motif,number_of_mutations_allowed):
    motif = arbitrary_motif(size_of_motif)
    motif_profile=get_profile(motif)
    list_of_list=lmer(set_of_sequences,size_of_motif)
    final_list=[]
    for inner_list in list_of_list:
        for el in inner_list:
            final_list.append(el)
    final_list=list(set(final_list))
    for l_mer in final_list:
        lmer_profile=get_profile(l_mer)
        for j in range(number_of_mutations_allowed+1)
            if entropy(lmer_profile,set_of_sequences, size_of_motif)>entropy(motif_profile,set_of_sequences, size_of_motif):
                motif = l_mer
                motif_profile=lmer_profile
            l_mer = best_neighbour(lmer_profile, l_mer)
            lmer_profile = get_profile(l_mer)
    print (motif)

#gets the arbitary motif
def arbitrary_motif(size_of_motif):
    valid_letters="ATGC"
    random_motif=''.join(random.choice(valid_letters)for i in range(size_of_motif))
    return random_motif
    # print(random_motif)
# arbitrary_motif(3)

#calculate the 4xl matrix which is the profile
def get_profile(sequence):
    length=len(sequence)
    rowi=['A','T','G','C']
    columni=list(sequence)
    row=[]
    column=[]
    for i in range(4):
        for j in range(length):
            if rowi[i]==columni[j]:
                column.append(0.50000)
            else:
                column.append(0.16667)
        row.append(column)
        column=[]
    return(row)

#gives a list of all the lmers
def lmer(set_of_sequences, ending_index):
    list3=[]
    for s in set_of_sequences:
        list2 = []
        a=0
        n_of_lmer= len(s) - ending_index + 1
        e = ending_index
        while(n_of_lmer):
            list1 = []
            for i in range(a, e):
                list1.append(s[i])
            str1 = ''.join(list1)
            list2.append(str1)
            n_of_lmer-=1
            a+=1
            e+=1
        list3.append(list2)
    #print(list3)
    return list3

#starts calculating the entropy to finaly compare them
def entropy(profile,set_of_sequences,size_of_motif):
    ent=0
    for s in set_of_sequences:
        ent+=entropy1(profile,s,size_of_motif)
    return ent
def entropy1(profile,sequence,k):
    start = 0
    end = start + k
    list_of_strings = []
    while (end != len(sequence) + 1):
        list_of_strings.append(sequence[start:end])
        start = start + 1
        end = start + k
    ent_list = []
    for p in list_of_strings:
        ent= entropy2(profile,p) #tc
        ent_list.append(ent)
    max_ent = ent_list[0]
    for ent in ent_list:
        if max_ent<ent:
            max_ent=ent
    return max_ent
def entropy2(profile,p):
    ent=0
    for i in range(len(p)):
        rowi = ['A', 'T', 'G', 'C']
        row_indx=rowi.index(p[i])
        # print(type(profile[row_indx][i]))
        ent+=math.log(profile[row_indx][i],math.e)
    return ent

#gives the best neighbour according to the profile
def best_neighbour(profile,lmer):
    rowi = ['A', 'T', 'G', 'C']
    column= list(lmer)

    neighbour_list = []
    sum_list = []

    for i in range(len(column)):
        if profile[0][i] <0.50000:
            profile[0][i]=0.55000
            i=i % 4
            column[i]=rowi[i]
            neighbour_list.append(''.join(column))
            column=list(lmer)
            for j in range(len(column)):
                if j!=i:
                   if profile[0][j]==0.50000:
                       profile[0][j]=0.27000
                   else:
                       profile[0][j]=0.09000
            sum_list.append(sum(profile[0]))
    max_sum=sum_list[0]
    for i in range(len(sum_list)):
        index=i
        if max_sum<sum_list[i]:
            max_sum=sum_list[i]
            index=i
    return neighbour_list[index]

profile_branching(['TGCTGC','TTTCG','ATTTAGC'],5,2)