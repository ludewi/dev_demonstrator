# Importieren der notwendigen Bibliotheken
import flwr as fl
import sys
import numpy as np
import streamlit as st
import time

# define the strategy object / strategy class
class SaveModelStrategy(fl.server.strategy.FedAvg): #FedAvg averages cell by cell all the model together
      # Minimum number of clients to be sampled for the next round
      # Minimum number of clients that need to be connected to the server before a training round can start
    
    def num_evaluation_clients(self, num_available_clients):
        st.write("Ich habe meine clients gefudnen")
        num_clients = int(num_available_clients * self.fraction_eval)
        st.write(num_clients)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def aggregate_fit(  #when I get the fit data, the training parameters from the clients
            self,
            rnd,
            results,
            failures ):
        with st.spinner("Ich aggregiere die Gewichte..."):
            st.write("Hallo")
            time.sleep(5)
            aggregated_weights = super().aggregate_fit(rnd, results, failures)
            #print(aggregated_weights)
            print(np.array(aggregated_weights))
            #st.write(aggregated_weights)
            if aggregated_weights is not None:
                # Save aggregated_weights
                print(f"Saving round {rnd} aggregated_weights...")
                np.savez(f"round-{rnd}-weights.npz", *aggregated_weights) # safe the data as a checkpoint
            st.success("Gewichte erfolgreiche aggregiert!")
        return aggregated_weights # return, because the aggregated weights will be used by flower

    
# Create strategy and run server
strategy = SaveModelStrategy(min_fit_clients=2,min_available_clients=2) # create an object of the class
#strategy = fl.server.strategy.FedAvg(
    #fraction_fit=0.1,  # Sample 10% of available clients for the next round
    
#)

try: 
    while True:
        # Start Flower server for three rounds of federated learning
        with st.spinner("Server läuft..."):
            fl.server.start_server( # takes 4 arguments
                server_address = 'localhost:8080' , #1 the place where to server entry and endpoint will open + Portnumber
                config={"num_rounds": 5} , # how many times we will call the clients for training, how many times we will aggregated weights from them
                grpc_max_message_length = 1024*1024*1024, # grpc sends commands, maximum message length, will carry the weights of the model, has to be long enough, if the model is small we can reduce it to save bandwith
                strategy = strategy #the strategy will be an object of the call that we have defined above
)
except KeyboardInterrupt:
    pass