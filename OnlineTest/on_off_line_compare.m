data_offline = rawEEG;
data_online_history = data_history;

y = 1:501;
plot(y, data_offline(1,(length(data_offline)-500):length(data_offline)), y, data_online_history(1,(length(data_online_history)-500):length(data_online_history)))
legend('offline','online');