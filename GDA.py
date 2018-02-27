import numpy as np
import matplotlib.pyplot as plt
import sys

class GDA:
    
    def __init__(self, input_file, output_file, threshold=0.0001, max_iters = 100000):
        self.x = self.ReadLinFile(input_file)
        self.y = self.ReadLinFile(output_file, "string")
        self.num_examples = self.x.shape[0]
        self.theta = np.zeros(3)
        self.threshold = threshold
        self.max_iterations = max_iters
    
    #function to read file with multiple feature
    def ReadLinFile(self, file_name, type="float"):
        fin = open(file_name, 'r')
        data = []
        if(type=="float"):
            for inp in fin:
                data.append(list(map(float,inp[:-1].split())))
        else:
            for inp in fin:
                data.append(inp[:-1])
        return np.array(data)
    
    #function to normalize data
    def NormalizeData(self):
        self.x = self.x.T
        mu1 = np.mean(self.x[0])
        mu2 = np.mean(self.x[1])
        sigma1 = np.std(self.x[0])
        sigma2 = np.std(self.x[1])
        self.x[0] = (self.x[0]-mu1)/sigma1
        self.x[1] = (self.x[1]-mu2)/sigma2
        self.x = self.x.T

    #function to compute mean vectors, covariance matrix, and Bernoulli parameter for linear separator
    def ComputeGDAParameters(self):
        self.class0x1, self.class0x2 = gda.x[np.where(gda.y=="Alaska")[0]].T
        self.class1x1, self.class1x2 = gda.x[np.where(gda.y=="Canada")[0]].T
        
        self.phi = self.class0x1.size/self.num_examples
        self.mu0 = np.array([np.mean(self.class0x1), np.mean(self.class0x2)])
        self.mu1 = np.array([np.mean(self.class1x1), np.mean(self.class1x2)])
        p = np.r_[np.c_[self.class0x1,self.class0x2]-self.mu0,np.c_[self.class1x1,self.class1x2]-self.mu1]
        self.sigma = np.dot(p.T,p)/self.num_examples
    
    #function to compute mean vectors, covariance matrices, and Bernoulli parameter for quadratic separator
    def ComputeGeneralGDAParameters(self):
        self.class0x1, self.class0x2 = gda.x[np.where(gda.y=="Alaska")[0]].T
        self.class1x1, self.class1x2 = gda.x[np.where(gda.y=="Canada")[0]].T
        
        self.phi = self.class0x1.size/self.num_examples
        self.mu0 = np.array([np.mean(self.class0x1), np.mean(self.class0x2)])
        self.mu1 = np.array([np.mean(self.class1x1), np.mean(self.class1x2)])
        p = np.c_[self.class0x1,self.class0x2]-self.mu0
        self.sigma0 = np.dot(p.T,p)/self.class0x1.shape[0]
        p = np.c_[self.class1x1,self.class1x2]-self.mu1
        self.sigma1 = np.dot(p.T,p)/self.class1x1.shape[0]
    
    #function to compute linear separator equation coefficients
    def SolveGDA(self):
        sigmainv = np.linalg.inv(self.sigma)
        self.theta[1:3] = np.dot(self.mu0.T - self.mu1.T, sigmainv)
        self.theta[0] = np.log((1-self.phi)/self.phi) - self.mu0.T.dot(sigmainv).dot(self.mu0)/2 + self.mu1.T.dot(sigmainv).dot(self.mu1)/2
    
#function to plot decision boundary along with input data
def plotBinaryClasses(gda, type):
    
    #plot input data points
    plt.scatter(gda.class0x1, gda.class0x2, label="Alaska", marker="+")
    plt.scatter(gda.class1x1, gda.class1x2, s=20, label="Canada", marker="x")
    
    a=50    #number of points in mesh in one dimension
    x=np.linspace(-15,15,a)
    theta = gda.theta
    if type == "scatterequation" or type == "scatterequationquadratic":
        #plot decision boundary 
        plt.plot(x, eval(str(theta[0]/(-theta[2])) +"+"+ str(theta[1]/(-theta[2]))+"*x"))
    #plot linear separator
    if type == "scatterequation":
        y = np.linspace(-2,2,a)
        X,Y = np.meshgrid(x,y)
        z0 = np.empty(a*a)
        meshthetas = np.c_[X.reshape((a*a,1)),Y.reshape((a*a,1))]
        sigmainv = np.linalg.inv(gda.sigma)
        
        #gaussian distribution for class 1
        for point in range(a*a):
            x1=meshthetas[point]
            z0[point] = np.exp(-1*(np.dot((x1-gda.mu0).dot(sigmainv),(x1-gda.mu0).T)))
        z0 = z0.reshape((a,a))
        
        #gaussian distribution for class 2
        z1 = np.empty(a*a)
        for point in range(a*a):
            x1=meshthetas[point]
            z1[point] = np.exp(-1*(x1-gda.mu1).dot(sigmainv).dot((x1-gda.mu1).T))
        z1 = z1.reshape((a,a))
        
        plt.contour(x,y,z0)
        plt.contour(x,y,z1)
    
    #plot quadratic separator
    elif type == "scatterequationquadratic":
        y = np.linspace(-15,15,a)
        X,Y = np.meshgrid(x,y)
        z = np.empty(a*a)
        meshthetas = np.c_[X.reshape((a*a,1)),Y.reshape((a*a,1))]
        sigma0inv = np.linalg.inv(gda.sigma0)
        sigma1inv = np.linalg.inv(gda.sigma1)
        for point in range(a*a):
            x1=meshthetas[point]
            z[point] = (x1-gda.mu1).T.dot(sigma1inv).dot(x1-gda.mu1) - (x1-gda.mu0).T.dot(sigma0inv).dot(x1-gda.mu0)
        z = z.reshape((a,a))
        
        C = -np.log(np.linalg.det(gda.sigma1)/np.linalg.det(gda.sigma0))
        
        
        plt.contour(x,y,z,[C])
    plt.legend()
    plt.title("GDA")
    plt.show()

if __name__=='__main__':
    
    #create a GDA object
    if(len(sys.argv)==3):
        gda = GDA(sys.argv[1],sys.argv[2])
    else:
        gda = GDA("q4x.dat","q4y.dat")
        
    #normalize data to 0 mean and 1 standard deviation
    gda.NormalizeData()
    
    #compute mean vectors, covariance matrix, and Bernoulli parameter for linear separator
    gda.ComputeGDAParameters()
    print("mu0: "+str(gda.mu0))
    print("mu1: "+str(gda.mu1))
    print("sigma:\n"+str(gda.sigma))
    print("phi: "+str(gda.phi))
    
    #plot input scatter data
    plotBinaryClasses(gda, "scatter")
    
    #compute parameters (coefficients) of the linear separator equation
    gda.SolveGDA()
    print(str(gda.theta[0])+" + "+str(gda.theta[1])+"x1 + "+str(gda.theta[2])+"x2 = 0")
    #plot linear separator
    plotBinaryClasses(gda, "scatterequation")
    
    #compute mean vectors, covariance matrix, and Bernoulli parameter for quadratic separator
    gda.ComputeGeneralGDAParameters()
    print("mu0: "+str(gda.mu0))
    print("mu1: "+str(gda.mu1))
    print("sigma0:\n"+str(gda.sigma0))
    print("sigma1:\n"+str(gda.sigma1))
    
    #plot quadratic separator using contour plot
    plotBinaryClasses(gda, "scatterequationquadratic")
    