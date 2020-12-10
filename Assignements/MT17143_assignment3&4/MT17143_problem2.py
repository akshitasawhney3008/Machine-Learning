#Graph and Fleury’s Algorithm
#MT17143
#AKSHITA SAWHNEY

class graph:
    def __init__(self,list_of_nodes,list_of_edges):
        self.list_of_nodes=list_of_nodes
        self.list_of_edges=list_of_edges

class node:
    def __init__(self,nodeid,degree):
        self.nodeid=nodeid
        self.degree=degree

class edge:
    def __init__(self, source, sink):
        self.source = source
        self.sink = sink

def degree_of_node(list_of_nodeob,list_of_edgeob):

    list_of_nodeob2=[]
    for i in range(len(list_of_nodeob)):
        degree = 0
        n1 = list_of_nodeob[i]
        nodei=n1.nodeid
        for j in range(len(list_of_edgeob)):
            if (list_of_edgeob[j].source==nodei or list_of_edgeob[j].sink==nodei):     #traverse the node in the list of edges if found increment the degree
                degree+=1
        n1.degree=degree
        list_of_nodeob2.append(n1)
    return (list_of_nodeob2)


def incident_edges(startnode,list_of_edgeob,list_of_nodeob):
    eulerianpath = []
    while len(list_of_edgeob) > 0:
        list_of_incident_edgeob=[]
        for i in range(len(list_of_edgeob)):
            if(startnode.nodeid==list_of_edgeob[i].source or startnode.nodeid==list_of_edgeob[i].sink):
                list_of_incident_edgeob.append(list_of_edgeob[i])           #keep a track of incident edges object for each node
        flag = 0
        for i in range(len(list_of_incident_edgeob)):
            bool=remove_edge(graph_ob1,startnode,list_of_incident_edgeob[i])     #remove the edge from the graph and then check if that edge is a branching or not
            if bool==True:
                eulerianpath.append(list_of_incident_edgeob[i])    #append the edge in the eulerian path
                flag=1
                for i in range(len(list_of_edgeob)):
                    if (startnode.nodeid == list_of_edgeob[i].source):
                        mysink = list_of_edgeob[i].sink
                        for i in range(len(list_of_nodeob1)):
                            if(list_of_nodeob1[i].nodeid==mysink):
                                startnode = list_of_nodeob1[i]
                                break
                        break
                    elif (startnode.nodeid == list_of_edgeob[i].sink):
                        mysource = list_of_edgeob[i].source
                        for i in range(len(list_of_nodeob1)):
                            if (list_of_nodeob1[i].nodeid == mysource):
                                startnode = list_of_nodeob1[i]
                                break
                        break
                list_of_edgeob.remove(list_of_incident_edgeob[i])
        if flag == 0:
            eulerianpath.append(list_of_incident_edgeob[0])
            for i in range(len(list_of_edgeob)):
                if (startnode.nodeid == list_of_edgeob[i].source):
                    mysink = list_of_edgeob[i].sink
                    for i in range(len(list_of_nodeob1)):
                        if (list_of_nodeob1[i].nodeid == mysink):
                            startnode = list_of_nodeob1[i]
                            break
                    break
                elif (startnode.nodeid == list_of_edgeob[i].sink):
                    mysource = list_of_edgeob[i].source
                    for i in range(len(list_of_nodeob1)):
                        if (list_of_nodeob1[i].nodeid == mysource):
                            startnode = list_of_nodeob1[i]
                            break
                    break
            list_of_edgeob.remove(list_of_incident_edgeob[0])
    return eulerianpath



def remove_edge(graph_ob1,startnode,incident_edgeob):
    list_of_edgeob2 = []
    for i in range(len(graph_ob1.list_of_edges)):
        list_of_edgeob2.append(graph_ob1.list_of_edges[i])
    list_of_nodeob3 = graph_ob1.list_of_nodes
    for i in range(len(graph_ob1.list_of_edges)):
        edge_ob1 = graph_ob1.list_of_edges[i]
        if (incident_edgeob.source == edge_ob1.source and incident_edgeob.sink == edge_ob1.sink):
            list_of_edgeob2.pop(i)                                           #remove the edge
        bool = BFS(graph_ob1, incident_edgeob, startnode,list_of_edgeob2)     #now calculate the BFS of the graph without the edge
        return bool

#BFS is calculate of the graph without the edge which has to be checked if it is branching or non_branching
#if we get all the nodes in the BFS traversal after removal of the edge then the edge that was removed will be non-branching edge
def BFS(graph_ob1, incident_edge_ob, startnode, list_of_edgeob2):
    flag = 0
    visited = []
    queue = [startnode]
    while (queue):
        node = queue.pop(0)
        for i in range(len(visited)):
            if (node.nodeid == visited[i].nodeid):
                flag = 1
            else:
               flag = 0
            if flag == 0:
                visited.append(node)
                for i in range(len(list_of_edgeob2)):
                    neighbours = []
                    print(node.nodeid, list_of_edgeob2[i].source, list_of_edgeob2[i].sink)
                    if (node.nodeid == list_of_edgeob2[i].source):
                        neighbours.append(list_of_edgeob2[i].sink)
                    elif(node.nodeid == list_of_edgeob2[i].sink):
                        neighbours.append(list_of_edgeob2[i].source)

                    for neighbour in neighbours:
                        for i in range(len(graph_ob.list_of_nodes)):
                            if (graph_ob.list_of_nodes[i].nodeid == neighbour):
                                queue.append(graph_ob.list_of_nodes[i])

                    # for i in range(len(graph_ob.list_of_nodes)):
                    #     for neighbour in neighbours:
                    #         if(graph_ob.list_of_nodes[i]==neighbour):
                    #             queue.append(graph_ob.list_of_nodes[i])



    if graph_ob.list_of_nodes == visited:
        return True
    else:
        return False
            # Get all edges incident on this node and put it in a list
            # Loop
            # flag = 0
                # Traverse this extracted edge list to get an edge each time
                # For this edge check BFS/remove edge
                # If this edge non bridge, remove and set starting node as the other node of this edge, flag = 1, break from loop
            # If flag == 0, remove a bridge


node_list=[]
list_of_edgeob=[]
list_of_nodeob=[]
list_of_nodeob1=[]
no_of_nodes=int(input("Enter the no_of_nodes"))
no_of_edges=int(input("Enter the no_of_edges"))
print("Enter the source sink pairs:")
for i in range(no_of_edges):
    n, m = map(int, input().strip().split(" "))
    node_list.append(n)
    node_list.append(m)
    e=edge(n,m)
    list_of_edgeob.append(e)                         #maintain a list of edges

node_list1=list(set(node_list))
for i in range(len(node_list1)):
    n=node(node_list1[i],0)
    list_of_nodeob.append(n)
graph_ob=graph(list_of_nodeob,list_of_edgeob)
list_of_nodeob1= degree_of_node(list_of_nodeob,list_of_edgeob)               #calculate the degree of each node
graph_ob1=graph(list_of_nodeob1,list_of_edgeob)
odd_degree=0
for i in range(len(list_of_nodeob1)):
    if(list_of_nodeob1[i].degree%2!=0):
        odd_degree+=1
if (odd_degree==0):
    startnode=list_of_nodeob1[0]
    ep = incident_edges(startnode,list_of_edgeob,list_of_nodeob1)
    for e in ep:
        print(e.source + ' ' + e.sink)
elif(odd_degree==2):
    for i in range(len(list_of_nodeob1)):
        if (list_of_nodeob1[i].degree % 2 != 0):
            startnode=list_of_nodeob1[i]
            ep = incident_edges(startnode,list_of_edgeob,list_of_nodeob1)
            for e in ep:
                print(str(e.source) + '-' + str(e.sink),end=' ')
else:
    print("eulerian path cannot be formed")