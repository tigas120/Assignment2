%% Load data and create train-test sets
df = csvread('wine.data');

Y = df(:,1) - 1;
X = df(:,2:end);
rng(4797);

train_test_partition = cvpartition(Y,'Holdout',0.2,'Stratify',true);
train_idx = training(train_test_partition);
test_idx = test(train_test_partition);
X_train = X(train_idx,:);
X_test = X(test_idx,:);
Y_train = Y(train_idx,:);
Y_test = Y(test_idx,:);

%% Train initial 0-vs-all model
Y_train_0_vs_all = Y_train;
Y_train_0_vs_all(Y_train_0_vs_all==0) = -1;
Y_train_0_vs_all(Y_train_0_vs_all~=-1) = 0;
Y_train_0_vs_all(Y_train_0_vs_all==-1) = 1;

opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 5;
ts_model_0_vs_all = genfis(X_train,Y_train_0_vs_all,opt);

%% Train initial 1-vs-all model
Y_train_1_vs_all = Y_train;
Y_train_1_vs_all(Y_train_1_vs_all==1) = -1;
Y_train_1_vs_all(Y_train_1_vs_all~=-1) = 0;
Y_train_1_vs_all(Y_train_1_vs_all==-1) = 1;

opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 5;
ts_model_1_vs_all = genfis(X_train,Y_train_1_vs_all,opt);

%% Train initial 2-vs-all model
Y_train_2_vs_all = Y_train;
Y_train_2_vs_all(Y_train_2_vs_all==2) = -1;
Y_train_2_vs_all(Y_train_2_vs_all~=-1) = 0;
Y_train_2_vs_all(Y_train_2_vs_all==-1) = 1;

opt = genfisOptions('FCMClustering','FISType','sugeno');
opt.NumClusters = 5;
ts_model_2_vs_all = genfis(X_train,Y_train_2_vs_all,opt);

%% Check initial performance on test set
Y_pred_init_0_vs_all = evalfis(ts_model_0_vs_all, X_test);
Y_pred_init_0_vs_all(Y_pred_init_0_vs_all<0) = 0;
Y_pred_init_0_vs_all(Y_pred_init_0_vs_all>1) = 1;

Y_pred_init_1_vs_all = evalfis(ts_model_1_vs_all, X_test);
Y_pred_init_1_vs_all(Y_pred_init_1_vs_all<0) = 0;
Y_pred_init_1_vs_all(Y_pred_init_1_vs_all>1) = 1;

Y_pred_init_2_vs_all = evalfis(ts_model_2_vs_all, X_test);
Y_pred_init_2_vs_all(Y_pred_init_2_vs_all<0) = 0;
Y_pred_init_2_vs_all(Y_pred_init_2_vs_all>1) = 1;

Y_pred_init = zeros(size(Y_test,1),3);
Y_pred_init(:,1) = Y_pred_init_0_vs_all + (1-Y_pred_init_1_vs_all) + (1-Y_pred_init_2_vs_all);
Y_pred_init(:,2) = Y_pred_init_1_vs_all + (1-Y_pred_init_0_vs_all) + (1-Y_pred_init_2_vs_all);
Y_pred_init(:,3) = Y_pred_init_2_vs_all + (1-Y_pred_init_0_vs_all) + (1-Y_pred_init_1_vs_all);
Y_pred_init = Y_pred_init./sum(Y_pred_init,2);
[~,Y_pred_init_labels] = max(Y_pred_init,[],2);
Y_pred_init_labels = Y_pred_init_labels - 1;

class_report_init = classperf(Y_test, Y_pred_init_labels);
fprintf('Initial Accuracy: %4.3f \n', class_report_init.CorrectRate);

%% Tune initial 0-vs-all model using ANFIS
[in,out,rule] = getTunableSettings(ts_model_0_vs_all);
anfis_model_0_vs_all = tunefis(ts_model_0_vs_all,[in;out],X_train,Y_train_0_vs_all,tunefisOptions("Method","anfis"));

%% Tune initial 1-vs-all model using ANFIS
[in,out,rule] = getTunableSettings(ts_model_1_vs_all);
anfis_model_1_vs_all = tunefis(ts_model_1_vs_all,[in;out],X_train,Y_train_1_vs_all,tunefisOptions("Method","anfis"));

%% Tune initial 2-vs-all model using ANFIS
[in,out,rule] = getTunableSettings(ts_model_2_vs_all);
anfis_model_2_vs_all = tunefis(ts_model_2_vs_all,[in;out],X_train,Y_train_2_vs_all,tunefisOptions("Method","anfis"));

%% Check ANFIS tuned model performance
Y_pred_final_0_vs_all = evalfis(anfis_model_0_vs_all, X_test);
Y_pred_final_0_vs_all(Y_pred_final_0_vs_all<0) = 0;
Y_pred_final_0_vs_all(Y_pred_final_0_vs_all>1) = 1;

Y_pred_final_1_vs_all = evalfis(anfis_model_1_vs_all, X_test);
Y_pred_final_1_vs_all(Y_pred_final_1_vs_all<0) = 0;
Y_pred_final_1_vs_all(Y_pred_final_1_vs_all>1) = 1;

Y_pred_final_2_vs_all = evalfis(anfis_model_2_vs_all, X_test);
Y_pred_final_2_vs_all(Y_pred_final_2_vs_all<0) = 0;
Y_pred_final_2_vs_all(Y_pred_final_2_vs_all>1) = 1;

Y_pred_final = zeros(size(Y_test,1),3);
Y_pred_final(:,1) = Y_pred_final_0_vs_all + (1-Y_pred_final_1_vs_all) + (1-Y_pred_final_2_vs_all);
Y_pred_final(:,2) = Y_pred_final_1_vs_all + (1-Y_pred_final_0_vs_all) + (1-Y_pred_final_2_vs_all);
Y_pred_final(:,3) = Y_pred_final_2_vs_all + (1-Y_pred_final_0_vs_all) + (1-Y_pred_final_1_vs_all);
Y_pred_final = Y_pred_final./sum(Y_pred_final,2);
[~,Y_pred_final_labels] = max(Y_pred_final,[],2);
Y_pred_final_labels = Y_pred_final_labels - 1;

class_report_final = classperf(Y_test, Y_pred_final_labels);
fprintf('Final Accuracy: %4.3f \n', class_report_final.CorrectRate);

%% Performance evaluation indexes
figure
cm=confusionchart(Y_test,Y_pred_final_labels);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = ' Confusion Matrix';
[m,order]=confusionmat(Y_test,Y_pred_final_labels);

Diagonal=diag(m);
sum_rows=sum(m,2);
Precision=Diagonal./sum_rows;
Overall_Precision=mean(Precision)
sum_col=sum(m,1);
recall=Diagonal./sum_col';
overall_recall=mean(recall)
F1_Score=2*((Overall_Precision*overall_recall)/(Overall_Precision+overall_recall))

%% Upload predicted scores to a file
csvwrite('predicted_anfis.txt',Y_pred_final_labels)