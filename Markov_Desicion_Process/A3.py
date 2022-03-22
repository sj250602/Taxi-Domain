import numpy as np
import random
import matplotlib.pyplot as plt
#actions = N,s,e,w,pu,pd
def reward_simulator(action,x_t,y_t,x_p,y_p,x_d,y_d):
    #choose action for given state because as given that the same action execution prob is 85%
    act = action
    if(action=="North"):    # for North
        list1 = ["North","South","East","West","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5,5,0, 0), k=1)[0] #Choosing the radnom action whith 85% prob for north and reaming for other
    if(action=="South"):
        list1 = ["South","North","East","West","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5,5,0, 0), k=1)[0]
    if(action=="East"):
        list1 = ["East","North","South","West","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5,5,0, 0), k=1)[0]
    if(action=="West"):
        list1 = ["West","North","South","East","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5,5,0, 0), k=1)[0]
    if(action=="Pickup"):
        list1 = ["Pickup","North","South","East","West","Putdown"]
        act = random.choices(list1, weights=(100, 0, 0, 0,0, 0), k=1)[0] 
    if(action=="Putdown"):
        list1 = ["Putdown","North","South","East","West","Pickup"]
        act = random.choices(list1, weights=(100, 0, 0, 0, 0, 0), k=1)[0]

    if(act=="Pickup"):
        if(x_p==x_t and y_p==y_t):
            return -1 # reward for pickup
        else:
            return -10 # reward when wrong pickup call
    elif(act=="Putdown"):
        if(x_p==x_t and y_p==y_t and (x_p==x_d and y_p==y_d)):
            return 20 # reach to destionation
        elif(x_p==x_t and y_p==y_t):
            return -1 # reward for putdown
        else:
            return -10 # wrong pickup call
    else:
        return -1 # reward for all other states

def simulator(action,x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,mp):
    act = action
    if(action=="North"):
        list1 = ["North","South","East","West","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5, 5, 0, 0), k=1)[0]
    if(action=="South"):
        list1 = ["South","North","East","West","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5, 5, 0, 0), k=1)[0]
    if(action=="East"):
        list1 = ["East","North","South","West","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5, 5, 0, 0), k=1)[0]
    if(action=="West"):
        list1 = ["West","North","South","East","Pickup","Putdown"]
        act = random.choices(list1, weights=(85, 5, 5, 5, 0, 0), k=1)[0]
    if(action=="Pickup"):
        list1 = ["Pickup","North","South","East","West","Putdown"]
        act = random.choices(list1, weights=(100, 0, 0, 0, 0, 0), k=1)[0]
    if(action=="Putdown"):
        list1 = ["Putdown","North","South","East","West","Pickup"]
        act = random.choices(list1, weights=(100, 0, 0, 0, 0, 0), k=1)[0]
    
    r = reward_simulator(act,x_t,y_t,x_p,y_p,x_d,y_d)
    if(act=="North"):
        if((x_t,y_t) in mp[act]): # check if it is able to go in nort or not 
            if(p_in_t):
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r # if wall or blocked return same value
            else:
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
        else:
            if(p_in_t and (x_t==x_p and y_t==y_p)):
                return x_t,y_t+1,x_p,y_p+1,x_d,y_d,p_in_t,r #reward( # if passenger in taxi and action is doable then return the next state
            else:
                return x_t,y_t+1,x_p,y_p,x_d,y_d,p_in_t,r # if passenger is not taxi and action is doable then return the next state
    if(act=="South"):
        if((x_t,y_t) in mp[act]):
            if(p_in_t):
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
            else:
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
        else:
            if(p_in_t and (x_t==x_p and y_t==y_p)):
                return x_t,y_t-1,x_p,y_p-1,x_d,y_d,p_in_t,r
            else:
                return x_t,y_t-1,x_p,y_p,x_d,y_d,p_in_t,r
    if(act=="East"):
        if((x_t,y_t) in mp[act]):
            if(p_in_t):
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
            else:
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
        else:
            if(p_in_t and (x_t==x_p and y_t==y_p)):
                return x_t+1,y_t,x_p+1,y_p,x_d,y_d,p_in_t,r
            else:
                return x_t+1,y_t,x_p,y_p,x_d,y_d,p_in_t,r
    if(act=="West"):
        if((x_t,y_t) in mp[act]):
            if(p_in_t):
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
            else:
                return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
        else:
            if(p_in_t and (x_t==x_p and y_t==y_p)):
                return x_t-1,y_t,x_p-1,y_p,x_d,y_d,p_in_t,r
            else:
                return x_t-1,y_t,x_p,y_p,x_d,y_d,p_in_t,r
    if(act=="Pickup"):
        if(x_t==x_p and y_t==y_p):
            return x_t,y_t,x_p,y_p,x_d,y_d,True,r
        else:
            return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
    if(act=="Putdown"):
        if(x_t==x_p and y_t==y_p):
            return x_t,y_t,x_p,y_p,x_d,y_d,False,r
        else:
            return x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r
    
def reward(action,state):
    if(action=="North" or action=="South" or action=="East" or action=="West"):
        return -1 # reward state for all action
    else:
        if(action=="Pickup"):
            if(state[0]!=state[1]):# when wrong pickup calls
                return -10
            else:
                return -1
        else:
            if(state[0]!=state[1]): # when wrong putdown calls
                return -10
            elif(state[0]!=state[3]):
                return -1
            else:
                return 20

def transition(action,state,gamma,U,mp):# calculate the Q(s,a)
    if(action=="North"):
        x_0,y_0 = state[0]
        x_1,y_1 = state[1]
        b = state[2]
        ans = 0
        if(b==True):
            if(state[0] in mp["East"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1+1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_1)+0.05*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1-1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_2)+0.05*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1+1),b,state[3]]
                ans += 0.85*reward(action,state_das_3)+0.85*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1-1),b,state[3]]
                ans += 0.05*reward(action,state_das_4)+0.05*gamma*U[str(state_das_4)]
        else:
            if(state[0] in mp["East"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_1)+0.05*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_2)+0.05*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1),b,state[3]]
                ans += 0.85*reward(action,state_das_3)+0.85*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_4)+0.05*gamma*U[str(state_das_4)]
        return ans
    if(action=="West"):
        x_0,y_0 = state[0]
        x_1,y_1 = state[1]
        b = state[2]
        ans = 0
        if(b==True):
            if(state[0] in mp["East"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1+1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_1)+0.05*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1-1,y_1),b,state[3]]
                ans += 0.85*reward(action,state_das_2)+0.85*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1+1),b,state[3]]
                ans += 0.05*reward(action,state_das_3)+0.05*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1-1),b,state[3]]
                ans += 0.05*reward(action,state_das_4)+0.05*gamma*U[str(state_das_4)]
        else:
            if(state[0] in mp["East"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_1)+0.05*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.85*reward(action,state_das_2)+0.85*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_3)+0.05*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_4)+0.05*gamma*U[str(state_das_4)]
        return ans
    if(action=="South"):
        x_0,y_0 = state[0]
        x_1,y_1 = state[1]
        b = state[2]
        ans = 0
        if(b==True):
            if(state[0] in mp["East"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1+1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_1)+0.05*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1-1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_2)+0.05*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1+1),b,state[3]]
                ans += 0.05*reward(action,state_das_3)+0.05*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1-1),b,state[3]]
                ans += 0.85*reward(action,state_das_4)+0.85*gamma*U[str(state_das_4)]
        else:
            if(state[0] in mp["East"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_1)+0.05*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_2)+0.05*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_3)+0.05*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1),b,state[3]]
                ans += 0.85*reward(action,state_das_4)+0.85*gamma*U[str(state_das_4)]
        return ans
    if(action=="East"):
        x_0,y_0 = state[0]
        x_1,y_1 = state[1]
        b = state[2]
        ans = 0
        if(b==True):
            if(state[0] in mp["East"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1+1,y_1),b,state[3]]
                ans += 0.85*reward(action,state_das_1)+0.85*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1-1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_2)+0.05*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1+1),b,state[3]]
                ans += 0.05*reward(action,state_das_3)+0.05*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1-1),b,state[3]]
                ans += 0.05*reward(action,state_das_4)+0.05*gamma*U[str(state_das_4)]
        else:
            if(state[0] in mp["East"]):
                ans += 0.85*reward(action,state)+0.85*gamma*U[str(state)]
            else:
                state_das_1 = [(x_0+1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.85*reward(action,state_das_1)+0.85*gamma*U[str(state_das_1)]
            if(state[0] in mp["West"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_2 = [(x_0-1,y_0),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_2)+0.05*gamma*U[str(state_das_2)]
            if(state[0] in mp["North"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_3 = [(x_0,y_0+1),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_3)+0.05*gamma*U[str(state_das_3)]
            if(state[0] in mp["South"]):
                ans += 0.05*reward(action,state)+0.05*gamma*U[str(state)]
            else:
                state_das_4 = [(x_0,y_0-1),(x_1,y_1),b,state[3]]
                ans += 0.05*reward(action,state_das_4)+0.05*gamma*U[str(state_das_4)]
        return ans
    if(action=="Pickup"):
        x_0,y_0 = state[0]
        x_1,y_1 = state[1]
        b = state[2]
        state_das = [(x_0,y_0),(x_1,y_1),b,state[3]]
        ans = reward(action,state_das)+gamma*U[str(state_das)]
        return ans
    if(action=="Putdown"):
        x_0,y_0 = state[0]
        x_1,y_1 = state[1]
        b = state[2]
        state_das = [(x_0,y_0),(x_1,y_1),b,state[3]]
        ans = reward(action,state_das)+gamma*U[str(state_das)]
        return ans
   
def value_iterations(gamma,epsilon,mp,list_actions,State): 
    
    norm = []
    iterations = 0
    change = 1000000.0
    U = {} # utility fnction
    U_das = {}
    policy = {} #policy values
    for s in State:
        U[str(s)] = 0.0
        U_das[str(s)] = 0.0
        policy[str(s)] = ""
    
    
    while(change>=(epsilon*(1-gamma)/gamma)):
        U = U_das.copy()
        #print(U)
        change = 0
        i = 0
        for s in State:
        # [(0,0),(1,4),F]->North = [(1,0)]*85% + [(0,1)]*0.05 + [(0,0)]
            list11 = []
            list12 = {}
            for act in list_actions:
                t = transition(act,s,gamma,U,mp)
                list11.append(t)
                list12[t] = act
            f = max(list11)
            U_das[str(s)] = f
            policy[str(s)] = list12[f]
            if(abs(U_das[str(s)]-U[str(s)])>change):
                change  = abs(U_das[str(s)]-U[str(s)])
                
        norm.append(change)
        iterations+=1
    iteration = [i for i in range(iterations)]
    return U,iteration,norm,policy


def Policy_Evalution(policy,U,gamma,mp,epsilon):
    norm = 0
    D = U.copy()
    change = 100000
    while(change>=(epsilon*(1-gamma)/gamma)):
        change = 0
        for s in State:
            v = U[str(s)]
            U[str(s)] = transition(policy[str(s)],s,gamma,U,mp)
            if(abs(U[str(s)]-v)>change):
                change  = abs(U[str(s)]-v)
    
    for s in D:
        norm += (D[s] - U[s])**2
    norm = norm**(1/2)
    return norm
def policy_iterations(gamma,epsilon,mp,list_actions,State): 
    
    norm = []
    iterations = 0
    change = False
    U = {}
    policy = {}
    for s in State:
        U[str(s)] = 0.0
        policy[str(s)] = "East"
    
    
    while(change!=True):
        change = True
        i = 0
        for s in State:
        # [(0,0),(1,4),F]->North = [(1,0)]*85% + [(0,1)]*0.05 + [(0,0)]
            v = policy[str(s)]
            list11 = []
            list12 = {}
            for act in list_actions:
                t = transition(act,s,gamma,U,mp)
                list11.append(t)
                list12[t] = act
            f = max(list11)
            policy[str(s)] = list12[f]
            if(v!=policy[str(s)]):
                change = False
        
        norm1 = Policy_Evalution(policy,U,gamma,mp,epsilon)
        norm.append(norm1)        
        iterations+=1
    iteration = [i for i in range(iterations)]
    return U,iteration,policy,norm

def part_b(gamma,epsilon,mp,list_actions,State,alpha,part): 
    
    dis_reward = []
    iterations = 1
    Q = {} # Q fnction
    Q_das = {}
    policy = {} #policy values
    for s in State:
        Q_das[str(s) + "West"] = 0.0
        Q_das[str(s) + "East"] = 0.0
        Q_das[str(s) + "South"] = 0.0
        Q_das[str(s) + "North"] = 0.0
        Q_das[str(s) + "Putdown"] = 0.0
        Q_das[str(s) + "Pickup"] = 0.0
        policy[str(s)] = "West"
    
    episode = 2000
    max_step = 500
    
    while(iterations!=episode):
        Q = Q_das.copy()
        #print(U)
        setp = 0
        change = 0
        s = random.choices(State)[0]
        fs = [(s[3]),(s[3]),False,(s[3])]
        while(s!=fs and setp<max_step):
            setp+=1
            if(part=="a" or part=="c"):
                l = random.uniform(0,1)
                if l<epsilon:
                    action = policy[str(s)]
                else:
                    action = random.choice(list_actions)
            elif(part=="b" or part=="d"):
                l = random.uniform(0,1)
                if l<epsilon/iterations:
                    action = policy[str(s)]
                else:
                    action = random.choice(list_actions)

            x_t,y_t = s[0]
            x_p,y_p = s[1]
            x_d,y_d = s[3]
            p_in_t = s[2]
            x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r = simulator(action,x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,mp)
            s_das = [(x_t,y_t),(x_p,y_p),p_in_t,(x_d,y_d)]
            change+=r*(gamma**(setp-1))
            if(part=="a" or part=="b"):   
                d = 0
                plcy = action
                if(str(s_das) + "West" in Q_das):
                    max_val = Q_das[str(s_das) + "West"]
                    d = 1
                    plcy = "West"
                if(str(s_das) + "East" in Q_das):
                    if(d==1):
                        if(Q_das[str(s_das) + "East"]>max_val):
                            max_val = Q_das[str(s_das) + "East"]
                            plcy = "East"
                    else:
                        max_val = Q_das[str(s_das) + "East"]
                        d = 1
                        plcy = "East"
                if(str(s_das) + "South" in Q_das):
                    if(d==1):
                        if(Q_das[str(s_das) + "South"]>max_val):
                            max_val = Q_das[str(s_das) + "South"]
                            plcy = "South"
                    else:
                        max_val = Q_das[str(s_das) + "South"]
                        d = 1
                        plcy = "South"
                if(str(s_das) + "North" in Q_das):
                    if(d==1):
                        if(Q_das[str(s_das) + "North"]>max_val):
                            max_val = Q_das[str(s_das) + "North"]
                            plcy = "North"
                    else:
                        max_val = Q_das[str(s_das) + "North"]
                        d = 1
                        plcy = "North"
                if(str(s_das) + "Pickup" in Q_das):
                    if(d==1):
                        if(Q_das[str(s_das) + "Pickup"]>max_val):
                            max_val = Q_das[str(s_das) + "Pickup"]
                            plcy = "Pickup"
                    else:
                        max_val = Q_das[str(s_das) + "Pickup"]
                        d = 1
                        plcy = "Pickup"
                if(str(s_das) + "Putdown" in Q_das):
                    if(d==1):
                        if(Q_das[str(s_das) + "Putdown"]>max_val):
                            max_val = Q_das[str(s_das) + "Putdown"]
                            plcy = "Putdown"
                    else:
                        max_val = Q_das[str(s_das) + "Putdown"]
                        d = 1
                        plcy = "Putdown"
                if(d==0):
                    max_val = 0
                
                sample = r + gamma*max_val
                Q_das[str(s)+action] = (1-alpha)*Q_das[str(s)+action]+alpha*sample
			
            if(part=="c"):
                l = random.uniform(0,1)
                if l<epsilon:
                    acton = policy[str(s)]
                else:
                    acton = random.choice(list_actions)
                
                plcy = acton
                sample = r + gamma*Q[str(s_das) +acton] 
                Q_das[str(s)+action] = (1-alpha)*Q_das[str(s)+action]+alpha*sample    
            elif(part=="d"):
                l = random.uniform(0,1)
                if l<epsilon/iterations:
                    acton = policy[str(s)]
                else:
                    acton = random.choice(list_actions)
                    
                plcy = acton    
                sample = r + gamma*Q[str(s_das) +acton] 
                Q_das[str(s)+action] = (1-alpha)*Q_das[str(s)+action]+alpha*sample 
            
            s = s_das
            policy[str(s)] = plcy
        
        dis_reward.append(change)
        iterations+=1
    iteration = [i for i in range(1,iterations)]
    return Q_das,iteration,dis_reward,policy

def part_last(gamma,epsilon,mp,list_actions,State,alpha): 
    
    dis_reward = []
    iterations = 1
    Q = {} # Q fnction
    Q_das = {}
    policy = {} #policy values
    for s in State:
        Q_das[str(s) + "West"] = 0.0
        Q_das[str(s) + "East"] = 0.0
        Q_das[str(s) + "South"] = 0.0
        Q_das[str(s) + "North"] = 0.0
        Q_das[str(s) + "Putdown"] = 0.0
        Q_das[str(s) + "Pickup"] = 0.0
        policy[str(s)] = "West"
    
    episode = 10000
    max_step = 500
    
    while(iterations!=episode):
        Q = Q_das.copy()
        #print(U)
        setp = 0
        change = 0
        s = random.choices(State)[0]
        fs = [(s[3]),(s[3]),False,(s[3])]
        while(s!=fs and setp<max_step):
            setp+=1
            l = random.uniform(0,1)
            if l<epsilon/iterations:
                action = policy[str(s)]
            else:
                action = random.choice(list_actions)

            x_t,y_t = s[0]
            x_p,y_p = s[1]
            x_d,y_d = s[3]
            p_in_t = s[2]
            x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,r = simulator(action,x_t,y_t,x_p,y_p,x_d,y_d,p_in_t,mp)
            s_das = [(x_t,y_t),(x_p,y_p),p_in_t,(x_d,y_d)]
            change+=r*(gamma**(setp-1))   
            d = 0
            plcy = action
            if(str(s_das) + "West" in Q_das):
                max_val = Q_das[str(s_das) + "West"]
                d = 1
                plcy = "West"
            if(str(s_das) + "East" in Q_das):
                if(d==1):
                    if(Q_das[str(s_das) + "East"]>max_val):
                        max_val = Q_das[str(s_das) + "East"]
                        plcy = "East"
                else:
                    max_val = Q_das[str(s_das) + "East"]
                    d = 1
                    plcy = "East"
            if(str(s_das) + "South" in Q_das):
                if(d==1):
                    if(Q_das[str(s_das) + "South"]>max_val):
                        max_val = Q_das[str(s_das) + "South"]
                        plcy = "South"
                else:
                    max_val = Q_das[str(s_das) + "South"]
                    d = 1
                    plcy = "South"
            if(str(s_das) + "North" in Q_das):
                if(d==1):
                    if(Q_das[str(s_das) + "North"]>max_val):
                        max_val = Q_das[str(s_das) + "North"]
                        plcy = "North"
                else:
                    max_val = Q_das[str(s_das) + "North"]
                    d = 1
                    plcy = "North"
            if(str(s_das) + "Pickup" in Q_das):
                if(d==1):
                    if(Q_das[str(s_das) + "Pickup"]>max_val):
                        max_val = Q_das[str(s_das) + "Pickup"]
                        plcy = "Pickup"
                else:
                    max_val = Q_das[str(s_das) + "Pickup"]
                    d = 1
                    plcy = "Pickup"
            if(str(s_das) + "Putdown" in Q_das):
                if(d==1):
                    if(Q_das[str(s_das) + "Putdown"]>max_val):
                        max_val = Q_das[str(s_das) + "Putdown"]
                        plcy = "Putdown"
                else:
                    max_val = Q_das[str(s_das) + "Putdown"]
                    d = 1
                    plcy = "Putdown"
            if(d==0):
                max_val = 0
            
            sample = r + gamma*max_val
            Q_das[str(s)+action] = (1-alpha)*Q_das[str(s)+action]+alpha*sample
        
            s = s_das
            policy[str(s)] = plcy
        
        dis_reward.append(change)
        iterations+=1
    iteration = [i for i in range(1,iterations)]
    return Q_das,iteration,dis_reward,policy

 
 
if __name__ == '__main__':
    mp = {"North":[],"South":[],"East":[],"West":[]}
    for i in range(5):
        mp["North"].append((i,4))
        mp["South"].append((i,0))
        mp["East"].append((4,i))
        mp["West"].append((0,i))
    mp["East"].append((0,0))
    mp["East"].append((0,1))
    mp["East"].append((1,3))
    mp["East"].append((1,4))
    mp["West"].append((2,3))
    mp["West"].append((2,4))
    mp["West"].append((1,0))
    mp["West"].append((1,1))
    mp["West"].append((3,0))
    mp["West"].append((3,1))
    mp["West"].append((2,0))
    mp["West"].append((2,1))
    
    State = []
    for i in range(5):
        for j in range(5):
            State.append([(i,j),(i,j),True,(0,0)])
            State.append([(i,j),(i,j),True,(0,4)])
            State.append([(i,j),(i,j),True,(4,4)])
            State.append([(i,j),(i,j),True,(3,0)])
    
    for i in range(5):
        for j in range(5):
            State.append([(i,j),(0,0),False,(0,4)])
            State.append([(i,j),(0,0),False,(4,4)])
            State.append([(i,j),(0,0),False,(3,0)])
            
            State.append([(i,j),(3,0),False,(4,4)])
            State.append([(i,j),(3,0),False,(0,0)])
            State.append([(i,j),(3,0),False,(0,4)])
            
            State.append([(i,j),(4,4),False,(3,0)])
            State.append([(i,j),(4,4),False,(0,0)])
            State.append([(i,j),(4,4),False,(0,4)])
            
            State.append([(i,j),(0,4),False,(0,0)])
            State.append([(i,j),(0,4),False,(3,0)])
            State.append([(i,j),(0,4),False,(4,4)])
    
    list_actions = ["North","South","East","West","Pickup","Putdown"]
    
    print("Enter 1 for Value Iteration ")
    print("Enter 2 for Policy Iteration")
    print("Enter 3 for Q function using constant epsilon")
    print("Enter 4 for Q function using decaying epsilon")
    print("Enter 5 for SARSA constant epsilon")
    print("Enter 6 for SARSA decaying epsilon")
    print("Enter 7 for last part")
    faltu_var = int(input("Enter the number : "))
    
    if(faltu_var==1):
        gamma = float(input("Enter the gamma(Discount Factor) : "))
        epsilon = float(input("Enter the epsilon(Exploration Rate) : "))
        Utility,Iteration,norm,opt_pol = value_iterations(gamma,epsilon,mp,list_actions,State)
        print("--------Value Iteration Data--------")
        print(f'Iterations = {len(Iteration)}')
        lst = []
        for s in State:
            lst.append(opt_pol[str(s)])
        #print(lst[:20])
        plt.plot(Iteration,np.array(norm))
        plt.xlabel('Iterations')
        plt.ylabel('Norm')
        plt.title(f'Norm vs Iterations,Gamma={gamma},Epsilon = {epsilon}')
        plt.show()
    elif(faltu_var==2):
        gamma = float(input("Enter the gamma(Discount Factor) : "))
        epsilon = float(input("Enter the epsilon(Exploration Rate) : "))
        Utility_pol,Iteration_pol,opt_pol_pol,norm_pol = policy_iterations(gamma,epsilon,mp,list_actions,State)
        print("--------Policy Iteration Data--------")
        print(f'Iterations = {len(Iteration_pol)}')
        #print(opt_pol_pol)
        plt.plot(Iteration_pol,np.array(norm_pol))
        plt.xlabel('Iterations')
        plt.ylabel('Policy Loss')
        plt.title(f'Policy Loss vs Iterations,Gamma={gamma},Epsilon = {epsilon}')
        plt.show()
    elif(faltu_var==7):
        mp1 = {"North":[],"South":[],"East":[],"West":[]}
        for i in range(10):
            mp1["North"].append((i,9))
            mp1["South"].append((i,0))
            mp1["East"].append((9,i))
            mp1["West"].append((0,i))
        mp1["East"].append((0,0))
        mp1["East"].append((0,1))
        mp1["East"].append((0,2))
        mp1["East"].append((0,3))
        mp1["West"].append((1,0))
        mp1["West"].append((1,1))
        mp1["West"].append((1,2))
        mp1["West"].append((1,3))
        mp1["East"].append((2,6))
        mp1["East"].append((2,7))
        mp1["East"].append((2,8))
        mp1["East"].append((2,9))
        mp1["West"].append((3,6))
        mp1["West"].append((3,7))
        mp1["West"].append((3,8))
        mp1["West"].append((3,9))
        mp1["East"].append((3,0))
        mp1["East"].append((3,1))
        mp1["East"].append((3,2))
        mp1["East"].append((3,3))
        mp1["West"].append((4,0))
        mp1["West"].append((4,1))
        mp1["West"].append((4,2))
        mp1["West"].append((4,3))
        mp1["East"].append((7,0))
        mp1["East"].append((7,1))
        mp1["East"].append((7,2))
        mp1["East"].append((7,3))
        mp1["West"].append((8,0))
        mp1["West"].append((8,1))
        mp1["West"].append((8,2))
        mp1["West"].append((8,3))
        mp1["East"].append((7,6))
        mp1["East"].append((7,7))
        mp1["East"].append((7,8))
        mp1["East"].append((7,9))
        mp1["West"].append((8,6))
        mp1["West"].append((8,7))
        mp1["West"].append((8,8))
        mp1["West"].append((8,9))        
        mp1["East"].append((5,4))
        mp1["East"].append((5,5))
        mp1["East"].append((5,6))
        mp1["East"].append((5,7))
        mp1["West"].append((6,4))
        mp1["West"].append((6,5))
        mp1["West"].append((6,6))
        mp1["West"].append((6,7))
        
        State1 = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    for l in range(10):
                        State1.append([(i,j),(k,l),True,(0,1)])
                        State1.append([(i,j),(k,l),True,(0,9)])
                        State1.append([(i,j),(k,l),True,(4,0)])
                        State1.append([(i,j),(k,l),True,(9,0)])
                        State1.append([(i,j),(k,l),True,(6,5)])
                        State1.append([(i,j),(k,l),True,(5,9)])
                        State1.append([(i,j),(k,l),True,(3,6)])
                        State1.append([(i,j),(k,l),True,(8,9)])
                        State1.append([(i,j),(k,l),False,(0,1)])
                        State1.append([(i,j),(k,l),False,(0,9)])
                        State1.append([(i,j),(k,l),False,(4,0)])
                        State1.append([(i,j),(k,l),False,(9,0)])
                        State1.append([(i,j),(k,l),False,(6,5)])
                        State1.append([(i,j),(k,l),False,(5,9)])
                        State1.append([(i,j),(k,l),False,(3,6)])
                        State1.append([(i,j),(k,l),False,(8,9)])
        gamma = float(input("Enter the gamma(Discount Factor) : "))
        epsilon = float(input("Enter the epsilon(Exploration Rate) : "))
        alpha = float(input("Enter the alpha(Learning Rate) : "))
        Q_table,Iteration_pol,norm_pol,opt_pol_pol = part_last(gamma,epsilon,mp1,list_actions,State1,alpha)
        
        #print(f'Iterations = {len(Iteration_pol)}')
        #print(norm_pol)
        # #print(opt_pol_pol)
        plt.plot(np.array(Iteration_pol),np.array(norm_pol))
        plt.xlabel('Episode')
        plt.ylabel('Sum of Discounted Reward')
        plt.title(f'Reward vs Episode,Gamma={gamma},Epsilon = {epsilon},Alpha={alpha}')
        plt.show()    
        
    else:
        State1 = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        State1.append([(i,j),(k,l),True,(0,0)])
                        State1.append([(i,j),(k,l),True,(0,4)])
                        State1.append([(i,j),(k,l),True,(4,4)])
                        State1.append([(i,j),(k,l),True,(3,0)])
                        State1.append([(i,j),(k,l),False,(0,0)])
                        State1.append([(i,j),(k,l),False,(0,4)])
                        State1.append([(i,j),(k,l),False,(4,4)])
                        State1.append([(i,j),(k,l),False,(3,0)])
        gamma = float(input("Enter the gamma(Discount Factor) : "))
        epsilon = float(input("Enter the epsilon(Exploration Rate) : "))
        alpha = float(input("Enter the alpha(Learning Rate) : "))
        if(faltu_var==3):
            part = "a"
            print("--------Q function using constant epsilon--------")
        elif(faltu_var==4):
            part = "b"
            print("--------Q function using decaying epsilon--------")
        elif(faltu_var==5):
            part = "c"
            print("--------SARSA constant epsilon--------")
        else:
            part = "d"
            print("--------SARSA decaying epsilon--------")
            
        print("Enter a for 1 run ")
        print("Enter b for 10 runs")
        x = input("Enter choice : ")
        
        Q_table,Iteration_pol,norm_pol,opt_pol_pol = part_b(gamma,epsilon,mp,list_actions,State1,alpha,part)
        if(x=="b"):
            for _ in range(9):
                Utility_,Iteration_,norm_,opt_pol_ = part_b(gamma,epsilon,mp,list_actions,State1,alpha,part)
                norm_pol = [norm_[i]+norm_pol[i] for i in range(len(norm_))]
            norm_pol = [norm_pol[i]/10 for i in range(len(norm_pol))]
        
        #print(f'Iterations = {len(Iteration_pol)}')
        #print(norm_pol)
        # #print(opt_pol_pol)
        plt.plot(np.array(Iteration_pol),np.array(norm_pol))
        plt.xlabel('Episode')
        plt.ylabel('Sum of Discounted Reward')
        plt.title(f'Reward vs Episode,Gamma={gamma},Epsilon = {epsilon},Alpha={alpha}')
        plt.show()    