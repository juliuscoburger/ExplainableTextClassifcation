import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from data_loading import load_data
from LRP import LRP, delete_most_important_word, delete_most_important_chars, integrated_gradient, encode
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import string


class Main():
    def __init__(self,dataset,supergroups,size,epochs,learning_rate):
        self.dataset=dataset
        self.data_name=dataset
        self.supergroups=supergroups
        self.size=size
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model=Model(self.dataset, self.size, supergroups=self.supergroups)
        self.model.to(self.device)
        #self.model.load_state_dict(torch.load("./agnews_40,0.0001.pt", map_location=torch.device('cpu')))

        if self.supergroups:
            trans = 'supergroups'
        else: trans=None
        self.train_loader, self.test_loader = load_data(dataset=self.dataset, transformation=trans)

    def training(self):
        loss_crit = nn.CrossEntropyLoss()
        # optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print("Starting training...")
        loss_hist = []
        self.acc_hist = []
        self.f1_hist = []
        self.train_preds = pd.DataFrame()
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            total = 0
            correct = 0
            f1=0

            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = torch.tensor(inputs), torch.tensor(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(inputs)
                loss = loss_crit(y_pred, labels.long())
                _, predicted = torch.max(y_pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
                labs=labels.long().clone().cpu().detach()
                preds=predicted.clone().cpu().detach()
                f1 += f1_score(labs, preds, average='weighted',sample_weight=None)*labels.size(0)                #  F1 IS CALCULATED FOR EACH BATCH

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 40 == 39:  # report every 40 batches
                    print('[%d, %5d] loss: %.3f ' % (epoch + 1, i + 1, running_loss / 10), end="")
                    print('accuracy: %d %%' % (100 * correct / total))
                    print('f1 score: %d %%' % (100 * f1/total))
                    # running_loss = 0.0

                if epoch==self.epochs-1: #only report final predictions
                    boop = np.transpose([np.array(labs, np.array(preds))])
                    self.train_preds=self.train_preds.append(pd.DataFrame(boop),ignore_index=True)



            loss_hist.append(running_loss / total)
            # print(len(loss_hist),'loss history length')
            self.acc_hist.append(100 * correct / total)  # accuracy history IN PERCENTAGE
            self.f1_hist.append(100 * f1/total)  # f1 score history

        torch.save(self.model.state_dict(), './trained_model.pt')
        self.training_acc = (100 * correct / total)
        self.training_f1 = (100 * f1/total)

        self.train_preds=self.train_preds.rename(columns={0:'y_true',1:'y_pred'})
        self.train_preds.to_csv('./training_predictions.csv')

        print('Finished training')

    def testing(self):
        print('Starting testing')
        self.test_preds=pd.DataFrame()
        correct = 0
        total = 0
        f1=0
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = torch.tensor(inputs), torch.tensor(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
                labs=labels.long().clone().cpu().detach()
                preds=predicted.clone().cpu().detach()
                f1 += f1_score(labs, preds, average='weighted',sample_weight=None)*labels.size(0)

                boop = np.transpose([np.array([labs, np.array(preds)])])
                self.test_preds=self.test_preds.append(pd.DataFrame(boop), ignore_index=True)

        self.testing_acc = 100 * correct / total
        self.testing_f1 = 100 * f1/total
        print('Accuracy of the network on the test inputs: %d %%' % (100 * correct / total))
        print('F1 score of the network on the test inputs: %d %%' % (100 * f1 / total))

        self.test_preds=self.test_preds.rename(columns={0:'y_true',1:'y_pred'})
        self.test_preds.to_csv('./testing_predictions.csv')
        
        
    def test_cost_confidence(self, thresh):
        print('Starting testing')
        self.test_preds=pd.DataFrame()
        correct = 0
        total = 0
        f1=0
        conf_correct = 0
        conf_total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = torch.tensor(inputs), torch.tensor(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                #print(probs)
                conf_total += len([1 for i in range(labels.size(0)) if probs[i] >= thresh]) 
                conf_correct += len([1 for i in range(labels.size(0)) if probs[i] >= thresh and predicted[i] == labels[i]])
                #print(conf_correct/conf_total)
                correct += (predicted == labels.long()).sum().item()
                labs=labels.long().clone().cpu().detach()
                preds=predicted.clone().cpu().detach()
                f1 += f1_score(labs, preds, average='weighted',sample_weight=None)*labels.size(0)

                boop = np.transpose([np.array([labs, np.array(preds)])])
                self.test_preds=self.test_preds.append(pd.DataFrame(boop), ignore_index=True)

        self.testing_acc = 100 * correct / total
        self.testing_f1 = 100 * f1/total
        print("finaly",conf_correct/conf_total)
        print("Min", (conf_total/total), (total-conf_total)*2)
        print('Accuracy of the network on the test inputs: %d %%' % (100 * correct / total))
        print('F1 score of the network on the test inputs: %d %%' % (100 * f1 / total))

        self.test_preds=self.test_preds.rename(columns={0:'y_true',1:'y_pred'})
        self.test_preds.to_csv('./testing_predictions.csv')

    def lrp_apply(self):
        agnews_label = ["World", "Sports", "Business", "Sci/Tech"]
        twentynews_label = [str(i) for i in range(20)]

        if self.data_name == "20Newsgroups":
            total_classes = 20
            label = twentynews_label
        else:
            total_classes = 4
            label = agnews_label

        f = open("out.html", "w")

        #txt_input = "obama is watching basketball on tv for money!"
        
        txt_input = "The Blog Confusion\",\"\\I hear that we have a new word - vlog.  The amount of confusion this will result\in should be terrifying.\\My appologies to Abbott and Costello...  I couldn't resist.\\Abbott: I say Blogs's on first, Vlogs's on second, and Blogosphere's on third.\\Costello: Is Blog the publisher?\\Abbott: Yes.\\Costello: Is Blog going to have the video too?\\Abbott: Yes.\\Costello: And you don't know the fellows' names?\\Abbott: Well I should.\\Costello: Well then Blogs publishing the story?\\Abbott: Yes.\\Costello: I mean the persons's name.\\Abbott: Blog.\\Costello: The guy on first.\\Abbott: Blog!\\Costello: The first publisher.\\Abbott: Blog.\\Costello: The guy writing...\\Abbott: Blogs the publisher!\ ...\\".lower()
        txt_input = txt_input[:1014]
        print(txt_input)
        
        
        enc = encode(txt_input)
        alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list('’') + list('\n')
        print(enc.shape)

        for class_nr in range(total_classes):
            R = LRP(self.model, txt_input, class_nr, total_classes)
            relevance = R[0].cpu().detach().numpy()[0]
            relevance = np.sum(relevance, axis=0)
            #relevances = integrated_gradient(self.model, txt_input, class_nr, encoded=False)
            #relevance = np.sum(relevances, axis=0)
            # print(relevance[:len(txt_input)])
            f.write("<p>Relevance output for class " + label[class_nr] + ":</p>")
            f.write("<p>")
            for c in range(1014):
                r = relevance[c]
                #print(alphabet[np.argmax(enc[0][:,c])], np.argmax(enc[0][:,c]))
                if r <= 0:
                    if np.min(relevance) != 0:
                        r1 = 255 - int(255 * (r / (np.min(relevance))))
                    else:
                        r1 = 255
                    r2 = 255
                else:
                    r1 = 255
                    if np.min(relevance) != 0:
                        r2 = 255 - int(255 * (r / (np.max(relevance))))
                    else:
                        r2 = 255

                chara = alphabet[np.argmax(enc[0][:,c])]
                #print(enc[0,0,c], chara)
                if chara == "a" and enc[0,0,c] != 1:
                    chara = " "
                f.write(
                    "<span style=\"background-color:rgb(" + str(r1) + ",255," + str(r2) + ")\";>" +  chara + "</span>")
            f.write("</p> <br>")
            
        f.close
        print("done")

    def lrp_deletion_test(self, del_function, x_i=None, most_i=True):
        #print('Starting LRP deletion test')
        if self.dataset=="20Newsgroups":
            total_classes = 20
            if self.supergroups:
                total_classes = 6
        if self.dataset=="AGNews":
            total_classes = 4
        
        correct = 0
        total = 0
        self.model.eval()
    
        for data in self.test_loader:
            inputs2 = []
            inputs, labels = data
            #inputs = inputs[:10]
            #labels = labels[:10]
            for i in range(len(inputs)):
                #d = decode(inputs[i])
                lab = labels.cpu().detach().numpy()
                R = LRP(self.model, inputs[i], lab[i], total_classes, encoded=True)
                relevance = R[0].cpu().detach().numpy()[0]
                relevance = np.sum(relevance, axis=0)
                #new_sentence = delete_most_important_chars(inputs[i], relevance, x=5, most=True)
                #new_sentence = delete_most_important_word(inputs[i], relevance, most=True)
                if x_i is None:
                    new_sentence = del_function(inputs[i], relevance, most=most_i)
                else:
                    new_sentence = del_function(inputs[i], relevance, x=x_i, most=most_i)
                inputs2.append(new_sentence)
            
            #print("lrp deletion done")
            inputs = np.array(inputs2)
            #inputs, labels = torch.as_tensor(inputs).float() , torch.tensor(labels)
            inputs = torch.as_tensor(inputs).float()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
        print('Accuracy of the network on the test inputs: %d %%' % (100 * correct / total))
        self.testing_acc=100 * correct / total
        return
    
    
    def ig_deletion_test(self, del_function, x_i=None, most_i=True):
        #print('Starting LRP deletion test')
        
        correct = 0
        total = 0
        self.model.eval()
    
        for data in self.test_loader:
            inputs2 = []
            inputs, labels = data
            for i in range(len(inputs)):
                #d = decode(inputs[i])
                lab = labels.cpu().detach().numpy()
                relevances = integrated_gradient(self.model, inputs[i], lab[i], encoded=True)
                relevance = np.sum(relevances, axis=0)
                #new_sentence = delete_most_important_chars(inputs[i], relevance, x=5, most=True)
                #new_sentence = delete_most_important_word(inputs[i], relevance, most=True)
                if x_i is None:
                    new_sentence = del_function(inputs[i], relevance, most=most_i)
                else:
                    new_sentence = del_function(inputs[i], relevance, x=x_i, most=most_i)
                inputs2.append(new_sentence) 
            
            #print("lrp deletion done")
            inputs = np.array(inputs2)
            #inputs, labels = torch.as_tensor(inputs).float() , torch.tensor(labels)
            inputs = torch.as_tensor(inputs).float()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
            break
        print('Accuracy of the network on the test inputs: %d %%' % (100 * correct / total))
        self.testing_acc=100 * correct / total
        return
    
    def acc_plot(self):
        import matplotlib.pyplot as plt
        eps = [i for i in range(1, self.epochs + 1)]

        acc=plt.plot(eps, self.acc_hist,label='accuracy')
        f1=plt.plot(eps,self.f1_hist,label='f1 score')
        plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.show()

        plt.savefig(
            '_'.join(str(elem).replace('.', ',') for elem in (self.data_name, self.supergroups, self.size, self.epochs, self.learning_rate)))
        plt.clf()

