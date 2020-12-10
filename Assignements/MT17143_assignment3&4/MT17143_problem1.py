#Shortest Superstring
#MT17143
#AKSHITA SAWHNEY

def shortest_superstring(k,str1):
    length=len(str1)
    if(length<=k):     #length of the string should not be less that the value of k if less return the string as it is
        print(str1)
        return
    start=0
    #find all the kmers of the given string
    end=start+k
    list_of_strings=[]
    while(end!=length+1):
        list_of_strings.append(str1[start:end])    #list of all the kmers of the given string
        start=start+1
        end=start+k
    #print(list_of_strings)
    for i in range(len(list_of_strings)-1):
        stra=list_of_strings[i]
        strb=list_of_strings[i+1]
        list_of_strings[i+1]=overlap(stra,strb)   #pick two strings one after another and find their overlap
        strc=list_of_strings[i+1]
    print(strc)

def overlap(stra,strb):
    if stra == 'GTAC' and strb == 'TACG':
        print('Hi')

    c=len(stra)-1
    index=0
    count1=0
    count=0
    i=0
    tempc=c
    tempi=i
    while(c>=0):
       # for i in range(len(strb)):
            tempc = c
            tempi = i
            flag = 0
            # print(strb[i])
            # print(stra[c])
            if (strb[i]==stra[c]):
                while((tempc!=len(stra))&(tempi!=len(strb))):     #loop to start moving forward for comparing the sequence
                    # print(strb[tempi])
                    # print(stra[tempc])
                    if(strb[tempi] == stra[tempc]):
                      tempc+=1
                      tempi+=1
                      count+=1
                    else:
                        flag = 1
                        break
                c -= 1
                if tempc==len(stra):
                    flag = 1
                if count1 < count:              #keep a track of count of the matches found and take the index of the string where count value is max
                    count1 = count
                    index = tempi
                    if flag == 1:
                        index = tempi-1
                    else:
                        index = tempi
                count = 0
                # flag = 0

            else:
                c-=1
                tempc = c
    # print(index)
    # print(stra[0:])
    # print(strb[index+1:])
    strc=stra[0:]+strb[index+1:]            #overlap the strings
    return strc
            # else:
            #     return 1;
            # c-=1;

k_mer=int(input("Enter the value of k"))
str1 = input(("Enter the string1"))
shortest_superstring(k_mer,str1)


