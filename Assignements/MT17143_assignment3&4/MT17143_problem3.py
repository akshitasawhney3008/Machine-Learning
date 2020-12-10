#De Bruijn Graph
#MT17143
#AKSHITA SAWHNEY

def de_bruijin(nunmber_of_strings,list_of_strings):
    list_reversecompliment=[]
    list_combined={}
    #print(list_of_strings)

    #reverse the string
    for i in range(len(list_of_strings)):
        stra= list_of_strings[i]
        strb=stra[::-1]
        str1 = ''
        #compliment the string
        for a in range(len(strb)):
            if strb[a]=='A':
               str1=str1+'T'
            elif strb[a]=='T':
               str1=str1+'A'
            elif strb[a]=='G':
               str1 = str1 + 'C'
            elif strb[a]=='C':
               str1= str1 + 'G'
            else:
                print("Not a DNA Sequence")
                exit(1)
        list_reversecompliment.append(str1)
    #print(list_reversecompliment)
    # for l in list_of_strings :
    #     list_reversecompliment.append(list_of_strings)
    list_combined=list_of_strings+list_reversecompliment           #union of the two lists with k+1mers and reverse compliment of these k+1mers
    #print(list_combined)
    list_combined=list(set(list_combined))            #distinct kmers in the list
    #print(list_combined)

    #find the kmers for each k+1mer
    kmer_list=[]
    kmer1 = ''
    kmer2 = ''
    kmer_string = ''
    for ele in (list_combined):
        str=ele
        length=len(str)
        start=0
        for i in range(start,length-1):
            kmer1 = kmer1+str[i]
        start+=1
        for i in range(start,length):
            kmer2= kmer2 + str[i]
        kmer_list.append(kmer1)
        kmer_list.append(kmer2)
        i=0
        kmer_string= "("+kmer_list[i]+","+kmer_list[i+1]+")"
        kmer1 = ''
        kmer2 = ''
        print(kmer_string)
        kmer_list=[]
    # new_list = list(Set(old_list))
    # list_combined=set(list_reversecompliment)
    # print(list_combined)

                # print("Not a DNA sequence")
                # # exit(1)

list_of_strings=[]
file = open('P3','r')                                                         #take input from a file
lin = file.readlines()
for l in lin:
    list_of_strings.append(l.strip())
# number_of_strings=int(input("Enter the number of K-mers that you will be entering:"))
# print("Enter the K-mers:")
# for i in range(number_of_strings):
#         strings = input();
#         list_of_strings.append(strings)
de_bruijin(len(list_of_strings),list_of_strings)