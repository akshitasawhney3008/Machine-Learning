#PATTERN BRANCHING ALGORITHM
#MT17143
#AKSHITA SAWHNEY

import random
def pattern_branching(set_of_sequences,size_of_motif,number_of_mutations_allowed):
    motif = arbitrary_motif(size_of_motif)
    list_of_list=lmer(set_of_sequences,size_of_motif)
    final_list=[]
    for inner_list in list_of_list:
        for el in inner_list:
            final_list.append(el)
    final_list=list(set(final_list))
    for l_mer in final_list:
        for j in range(number_of_mutations_allowed+1):
            if find_distance(lmer(set_of_sequences, size_of_motif),l_mer)<find_distance(lmer(set_of_sequences, size_of_motif),motif):
                motif = l_mer
            l_mer = get_best_neighbour(set_of_sequences,neighbours(l_mer),size_of_motif,l_mer)
    print (motif)

#finds the arbitrary motif
def arbitrary_motif(size_of_motif):
    valid_letters="ATGC"
    random_motif=''.join(random.choice(valid_letters)for i in range(size_of_motif))
    return random_motif
    # print(random_motif)
# arbitrary_motif(3)

#finds the distance
def find_distance(set_of_patterns,sequence):

    list_of_dist1=[]
    for p in set_of_patterns:
        list_of_dist = []
        for pi in p:
            dist = 0
            for i in range(len(pi)):
                if(pi[i]!=sequence[i]):                #uses hamming distance to calculate the distance
                    dist+=1
            list_of_dist.append(dist)
        list_of_dist1.append(list_of_dist)

    list_of_min_dist=[]
    for l in(list_of_dist1):
        min_dist = l[0]                                                   #finds the minimum distances
        for j in range(1,len(l)):
            if(l[j]<min_dist):
                min_dist=l[j]
        list_of_min_dist.append(min_dist)
    # print(list_of_dist1)
    # print(list_of_min_dist)
    sum = 0
    for e in list_of_min_dist:                                                   #finds the sum of all the minimum distance
        sum += e
    return sum

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

# lmer(['ATTCGC','ATTCG','ATTCGAT'],3)

# gives the bestneighbour by taking all the neighbours and finding each of their distance as above
def get_best_neighbour(list_of_S,list_of_neighbours,size_of_motif,l_mer):
    minimum_distance=(find_distance(lmer(list_of_S, size_of_motif),list_of_neighbours[0]))
    best_neighbour=list_of_neighbours[0]
    for n in list_of_neighbours:
        dist=find_distance(lmer(list_of_S, size_of_motif),n)
        if dist<minimum_distance:
            minimum_distance=dist
            best_neighbour=n
    return best_neighbour

#returns a list of all the neighbours to a sequence with hamming distance 1
def neighbours(sequence):
    list_seq=[]
    sequence=list(sequence)
    for i in range(len(sequence)):
        if(sequence[i]=='A'):
            sequence[i] = 'T'
            list_seq.append(''.join(sequence))
            sequence[i] = 'G'
            list_seq.append(''.join(sequence))
            sequence[i] = 'C'
            list_seq.append(''.join(sequence))
            sequence[i] = 'A'
        if (sequence[i] == 'T'):
            sequence[i] = 'A'
            list_seq.append(''.join(sequence))
            sequence[i] = 'G'
            list_seq.append(''.join(sequence))
            sequence[i] = 'C'
            list_seq.append(''.join(sequence))
            sequence[i] = 'T'
        if (sequence[i] == 'G'):
            sequence[i] = 'T'
            list_seq.append(''.join(sequence))
            sequence[i] = 'A'
            list_seq.append(''.join(sequence))
            sequence[i] = 'C'
            list_seq.append(''.join(sequence))
            sequence[i] = 'G'
        if (sequence[i] == 'C'):
            sequence[i] = 'T'
            list_seq.append(''.join(sequence))
            sequence[i] = 'G'
            list_seq.append(''.join(sequence))
            sequence[i] = 'A'
            list_seq.append(''.join(sequence))
            sequence[i] = 'C'
    return list_seq
# neighbours(list('ATT'))

# print(get_best_neighbour(['ATTCGC','ATTCG','ATTCGAT'],['TTT', 'GTT', 'CTT', 'AAT', 'AGT', 'ACT', 'ATA', 'ATG', 'ATC'],3,'CCC'))

pattern_branching(['ATTCGC','ATTCG','ATTCGAT'],3,2)



















# set_of_sequences=input("Enter the initial set of sequences")
# size_of_motif=int(input("Enter the size of motif"))
# number_of_mutations_allowed=int(input("Enter the number of mutations allowed"))
# pattern_branching(set_of_sequences,size_of_motif,number_of_mutations_allowed)