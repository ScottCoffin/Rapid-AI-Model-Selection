#################################### Module1: Load needed packages and the data set in scope ########
library(shiny)
library(shinyBS)
library(shinythemes)
library(modelgrid)
library(magrittr)
library(caret)
library(e1071)
library(foreach)
library(recipes)
library(purrr)
library(dplyr)
library(pROC)
library(BradleyTerry2)
library(corrplot)
library(DataExplorer)
library(randomForest)
library(glmnet)
library(xgboost)
library(readr)
library(ggplot2)
library(mlbench)
library(e1071)
library(DALEX)
library(pdp)
library(ICEbox)
library(iml)
library(ingredients)
library(iBreakDown)
library(palmerpenguins)
library(shinycssloaders)
library(tidyverse)

####################
# load data
set.seed(9650)

data("penguins")
# load default dataset (palmers penguins
data_set <- penguins %>% 
  drop_na(sex) %>% 
  # code to demonstrate how the app handles missingness using multiple imputation
  mutate(across(-all_of("sex"), 
                ~ ifelse(runif(length(.)) < 0.05, NA, .)))  # 10% chance to replace with NA

######################## Module2:  pre-Processing the data using recipe
set.seed(9650)
Attri_rec_2 <- 
  recipe(sex ~ ., data = data_set) %>%        # Formula.
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>%               # convert nominal data into one or more numeric.
  step_impute_knn(all_predictors()) %>%           # impute missing data using nearest neighbors.
  step_zv(all_predictors()) %>%                 # remove variables that are highly sparse and unbalanced.
  step_corr(all_predictors()) %>%               # remove variables that have large absolute correlations with other variables.
  step_center(all_predictors()) %>%             # normalize numeric data to have a mean of zero.
  step_scale(all_predictors()) %>%              # normalize numeric data to have a standard deviation of one.
  prep(training = data_set, retain = TRUE)      # train the data recipe 

data_rec_2 <- as.data.frame(juice(Attri_rec_2))
# add the response variable
data_rec_2$sex <- data_set$sex
# str(data_rec_2)
# Create a Validation Dataset (training data 70% and validation data 30% of the original data)
set.seed(9650)
validation_index <- createDataPartition(data_rec_2$sex,times= 1,  p= 0.70, list=FALSE)
# select 30% of the data for validation
data_validation <- data_rec_2[-validation_index,]
# use the remaining 70% of data to train the models
data_train <- data_rec_2[validation_index, ]
# For traing the models  
x_train <- data_train %>% dplyr::select(-sex) # Predictors
y_train <- data_train[["sex"]] # Response
# for validation/test
x_validation <- data_validation %>% dplyr::select(-sex)
y_validation <- data_validation[["sex"]]
## Vaiable list
variable_list <- colnames(x_validation)

##############  Module 3 Construct model grid and define shared settings.####################
set.seed(9650)
mg <- 
  model_grid() %>%
  share_settings(
    y = y_train,
    x = x_train,
    metric = "ROC",
    trControl = trainControl(
      method = "adaptive_cv",
      number = 10, repeats = 5,
      adaptive = list(min =3, alpha =0.05, method = "BT", complete = FALSE),
      search = "random",
      summaryFunction = twoClassSummary,
      classProbs = TRUE))
purrr::map_chr(mg$shared_settings, class)
## add models to train
mg_final <- mg %>%
  
  add_model(model_name = "LDA",
            method = "lda",
            custom_control = list(method = "repeatedcv",
                                  number = 10,
                                  repeats =5))%>%
  add_model(model_name = "eXtreme Gradient Boosting",method = "xgbDART")%>% 
  add_model(model_name = "Neural Network", method = "nnet")%>%
  add_model(model_name = "glmnet", method = "glmnet")%>%
  add_model(model_name = "Random Forest", method = "rf")
#%>% add_model(model_name = "gbm",
# method="gbm", custom_control = list(verbose = FALSE))

list_model <- c(names(mg_final$models))

################################# Module4: Shiny application #########################
ui <- fluidPage(theme = shinytheme("yeti"),
                tags$head(tags$link(rel = "stylesheet", type = "text/css", href = "styles.css")),
                ## Application title
                titlePanel(wellPanel("Rapid AI Builder App")),
                tags$div("Contact: Scott.Coffin@oehha.ca.gov", align = 'center', style = 'font-size: 15px; display: block; margin-left: auto; margin-right: auto;'), 
                navbarPage("Workflow ===>",
                           tabPanel("Exploratory Data Analysis",
                                    tabsetPanel(type = "tabs",
                                                tabPanel("Explore the data object",
                                                         navlistPanel(
                                                           tabPanel("1- Structure of the data object", verbatimTextOutput("str_ucture")),
                                                           tabPanel("2- Missing values?", plotOutput("missing_value"),
                                                                    br(),
                                                                    p("Any missing values will be imputed using the K-Nearest Neighbors approach"),
                                                                    ),
                                                           tabPanel("3- Correlation analysis", plotOutput("coor_plot", height = 700),
                                                                    br(),
                                                                    p("Missing tiles in this plot indicate that the column has NA values")
                                                                    ),
                                                           )
                                                         ),
                                                tabPanel("After Data Cleaning (using recipe method)",
                                                         navlistPanel(
                                                           tabPanel("1- Structure of the data object after Processing", 
                                                                    verbatimTextOutput("str_ucture_after")),
                                                           tabPanel("2- Missing values?", plotOutput("missing_value_after") ),
                                                           tabPanel("3- Correlation analysis after", plotOutput("coor_plot_after", height = 600))                                                         )))),
                           tabPanel("Fitting & Validation & Statistics",
                                    sidebarLayout(
                                      sidebarPanel(
                                        wellPanel(selectInput(inputId = "abd", 
                                                              label = "Choose Model : ",
                                                              choices = c("none", list_model),
                                                              selected = "none", width = '200px'))),
                                      mainPanel(
                                        tabsetPanel(type= "tabs",
                                                    ## output model training and summary
                                                    tabPanel("Model Training & Summary",
                                                             navlistPanel(
                                                               tabPanel("1- Show info model grid ",verbatimTextOutput("info_modelgrid")),
                                                               tabPanel("2- Performance statistics of the model grid (dotplot) ", 
                                                                        withSpinner(plotOutput("dotplot", width = 600, height = 600))),
                                                               tabPanel("3- Extract Performance of the model grid ", 
                                                                        withSpinner(verbatimTextOutput(outputId = "summary"))
                                                                        ),
                                                               tabPanel("4- Show the AUC & Accuracy of individual models (on data training)",
                                                                        withSpinner(verbatimTextOutput("Accurac_AUC")), 
                                                                        withSpinner(htmlOutput("best_model_train"))
                                                                        ),
                                                               tabPanel("5- Show Statistics of individual model", verbatimTextOutput(outputId = "Indiv_Analysis")
                                                                        ,"Examine the relationship between the estimates of performance 
                      and the tuning parameters", br(),
                                                                        withSpinner(plotOutput(outputId= "tuning_parameter"))),
                                                               tabPanel("6- Show single model's Accuracy (on data training)",
                                                                        verbatimTextOutput(outputId = "accuracy")))),
                                                    ## output model validation and statistics      
                                                    tabPanel("Model Validation & Statistics", 
                                                             navlistPanel(
                                                               tabPanel("1-Show the AUC & Accuracy of individual models (on validation data)", 
                                                                        verbatimTextOutput(outputId = "AUC_of_individ"), htmlOutput("best_model_vali")),
                                                               tabPanel("2- Show single model's Accuracy/data validation",
                                                                        verbatimTextOutput("accuracy_vali"))))
                                                    
                                        )))),
                           ###########################################################  Module post hoch explanation ################  
                           tabPanel("Model Explanation",
                                    sidebarLayout(
                                      sidebarPanel(
                                        wellPanel(selectInput(inputId = "perf", 
                                                              label = "Choose Model: ",
                                                              choices = c("none", list_model),
                                                              selected = "none", width = '200px')),
                                        
                                        wellPanel(selectInput(inputId = "Variab_le", 
                                                              label = "Choose Variable: ",
                                                              choices = c("none", variable_list),
                                                              selected = "none", width =  '200px')),
                                        wellPanel(numericInput(inputId="row_index", 
                                                               label= "Insert the row number", 
                                                               value=1,
                                                               min=1, max=20,  width = '200px'))),
                                      mainPanel(
                                        tabsetPanel(type = "tabs",
                                                    #### IML Packge#######
                                                    tabPanel("IML Package", 
                                                             tabsetPanel(type = "tabs",
                                                                         tabPanel("Global Interpretation",
                                                                                  navlistPanel(
                                                                                    tabPanel("1- Plot variable importance", plotOutput("variable_imp", height = "600px"), 
                                                                                             verbatimTextOutput("condition")),
                                                                                    tabPanel("2- Effect of a feature on predictions", 
                                                                                             plotOutput("PD_P", height = "600px"), htmlOutput("des")),
                                                                                    tabPanel("3- Feature Interaction", plotOutput("inter_action", height = "600px")),
                                                                                    tabPanel("4- 2-way interactions", 
                                                                                             plotOutput("two_way_inter", height = "600px"), 
                                                                                             verbatimTextOutput("condition2")),
                                                                                    tabPanel("5- ICE + PDP plots", plotOutput("IC_E")),
                                                                                    tabPanel("6- Global Surrogate model", 
                                                                                             plotOutput("surrogate_model", height = "600px")))),
                                                                         tabPanel("Local Interpretation",
                                                                                  navlistPanel(
                                                                                    tabPanel("1- Shapley Values", plotOutput(outputId = "shapley_Values", height = "600px")),
                                                                                    tabPanel("2- LIME", plotOutput(outputId = "single_plot_lime", height = "600px")))))),
                                                    #### DALEX Packge#######          
                                                    tabPanel("DALEX Package",
                                                             tabsetPanel(type = "tabs",
                                                                         tabPanel("Global Interpretation ",
                                                                                  navlistPanel(
                                                                                    tabPanel("1- Plot variable importance", 
                                                                                             plotOutput("variable_imp_da", height = "600px"), 
                                                                                             verbatimTextOutput("condition_da")),
                                                                                    tabPanel("2- Variable response", plotOutput("Vari_resp", height = "600px")),
                                                                                    tabPanel("3- Accumulated Local Effect (ALE)", plotOutput("ALE_da"), 
                                                                                             verbatimTextOutput("condition2_da")),
                                                                                    tabPanel("4- What-If Profiles/ICE analysis ", 
                                                                                             plotOutput("what_if", height = "600px")))),
                                                                         
                                                                         tabPanel("Local Interpretation ",
                                                                                  navlistPanel(
                                                                                    tabPanel("1- Shapley Values", plotOutput(outputId = "shapley_Values_da", height = "600px"), 
                                                                                             htmlOutput("descri_ption_sharp")),
                                                                                    tabPanel("2- BreakDown", plotOutput(outputId = "single_plot_brd_da", 
                                                                                                                         height = "600px"),htmlOutput("descri_ption")))),
                                                                         tabPanel("ModelDown Analysis", br(), br(), 
                                                                                  actionButton("do", "Click here to generate an HTML file with
                                                                                  the following summary information: auditor, drifter, model_performance, 
                                                                                               variable_importance, and variable_response", width = '90px' ))))
                                        ))))),          
                ############ Contact ############# 
                br(), br(),br(),
                h3("Rapid AI Model Builder App"),
                h4(
                   "by Scott Coffin, Ph.D.",
                   br(),
                   "Based on code originally developed by: Abderrahim Lyoubi-Idrissi", 
                   br(),
                   "Code available at", a("GitHub", href='https://github.com/ScottCoffin/Rapid-AI-Model-Selection.git'),
                   br()
                   )
)

################################# Define server logic #########################

server <- function(input, output, session) {
  
  set.seed(9650)    
  # before processing the data
  # data structure
  output$str_ucture <- renderPrint({
   str(data_set)
   # skimr::skim(data_set)
    
  })
  
  ## Missing values ? 
  output$missing_value <- renderPlot({
    plot_missing(data_set,
                 ggtheme = theme_minimal(base_size = 24))
  })
  # plot to check the results of step_corr(all_predictors())
  
  output$coor_plot <- renderPlot({
    plot_correlation(data_set,
                     ggtheme = theme_minimal(base_size = 24))
  })
  
  ## after processing the data 
  # data structure
  output$str_ucture_after <- renderPrint({
    str(data_rec_2)
  })
  # plot to check the missing values
  output$missing_value_after <- renderPlot({
    plot_missing(data_rec_2,
                 ggtheme = theme_minimal(base_size = 24))
  })
  # plot to check the correlation
  output$coor_plot_after <- renderPlot({
    plot_correlation(data_rec_2,
                     ggtheme = theme_minimal(base_size = 24))
  })
  set.seed(9650)
  ## Infos about the componets of the constructed model grid     
  output$info_modelgrid <- renderPrint({
    mg_final$models
  })
  ## train all models 
  set.seed(9650)
  mg_final_tra <- reactive({caret::train(mg_final)})
  ## plot to compare
  output$dotplot <- renderPlot({ 
    
    if (is.null(mg_final_tra()$model_fits)) {
      return("No models have been trained.")
    }
    
    mg_final_tra()$model_fits %>%
      caret::resamples(.) %>%
      lattice::dotplot(.,
                       par.settings = list(
                         axis.text = list(cex = 1.5),  # Increase size of axis text
                         par.xlab.text = list(cex = 1.8),  # Increase size of x-axis label
                         par.ylab.text = list(cex = 1.8),  # Increase size of y-axis label
                         par.main.text = list(cex = 2)  # Increase size of the main title
                       )
                       ) 
  })
  ## Show the overall summary
  set.seed(9650)
  output$summary <- renderPrint({
    mg_final_tra()$model_fits %>%
      caret::resamples(.) %>%
      summary(.)
  })
  
  ## computing the auc & the accuracy (on train data)
  set.seed(9650)
  output$Accurac_AUC <- renderPrint({
    AUC1 <- mg_final_tra()$model_fits%>% predict(., newdata = x_train, type ="prob")%>%
      map(~roc(y_train, .x[,2]))
    auc_value_train <- map_dbl(AUC1, ~(.x)$auc)
    auc_value_df_train_df <- as.data.frame(auc_value_train)
    accuarcy_all_train <- predict(mg_final_tra()$model_fits, newdata = x_train)%>% 
      map( ~confusionMatrix(.x, y_train))%>%
      map_dfr(~ cbind(as.data.frame(t(.x$overall)),as.data.frame(t(.x$byClass))), .id = "Modelname")%>%
      select(Modelname, Accuracy, Kappa)
    accuracy_auc_train <-  bind_cols(accuarcy_all_train, auc_value_df_train_df)
    accuracy_auc_train
  })
  ## computing the auc and the Accuracy
  set.seed(9650)
  output$best_model_train <- renderUI({
    
    AUC1 <- mg_final_tra()$model_fits%>% predict(., newdata = x_train, type ="prob")%>%
      map(~roc(y_train, .x[,2])) 
    
    auc_value_train <- map_dbl(AUC1, ~(.x)$auc)
    auc_value_df_train_df <- as.data.frame(auc_value_train)
    
    accuarcy_all_train <- predict(mg_final_tra()$model_fits, newdata = x_train)%>% 
      map( ~confusionMatrix(.x, y_train))%>%
      map_dfr(~ cbind(as.data.frame(t(.x$overall)),as.data.frame(t(.x$byClass))), .id = "Modelname")%>%
      select(Modelname, Accuracy, Kappa)
    
    accuracy_auc_train <-  bind_cols(accuarcy_all_train, auc_value_df_train_df)
    
    max_auc_train <- filter(accuracy_auc_train, auc_value_train == max(accuracy_auc_train$auc_value_train))%>%
      select(Modelname)
    max_Accuracy_train <- filter(accuracy_auc_train, Accuracy == max(accuracy_auc_train$Accuracy))%>%
      select(Modelname)
    
    HTML(paste("Results", br(),  
               "1- The model", strong(max_auc_train), "has the highest AUC value.", br(),
               "2- The model",strong(max_Accuracy_train), "has the highest Accuracy" ))
  })
  # Show the summary of individual model    
  output$Indiv_Analysis <- renderPrint({if(input$abd == "none"){print("Please select a model")}
    mg_final_tra()$model_fits[[input$abd]]
  })
  # Plot the individual model    
  output$tuning_parameter <- renderPlot({
    ggplot( mg_final_tra()$model_fits[[input$abd]]) +
      theme_minimal(base_size = 24)
  })
  
  # ConfusionMatrix on training data        
  output$accuracy <- renderPrint({if(input$abd == "none"){print("Please select a model")}
    else{ confusionMatrix( predict(mg_final_tra()$model_fits[[input$abd]], x_train), y_train)}
  })
  
  ########################### Model Validation #########################
  # Extract the auc values
  output$AUC_of_individ <- renderPrint({
    AUC2 <- mg_final_tra()$model_fits%>% predict(., newdata = x_validation, type ="prob")%>% 
      map(~roc(y_validation, .x[,2]))
    
    auc_value_vali <- map_dbl(AUC2, ~(.x)$auc)
    auc_value_df_vali_df <- as.data.frame(auc_value_vali)
    
    cf_vali <- predict(mg_final_tra()$model_fits, newdata = x_validation)%>% 
      map( ~confusionMatrix(.x, y_validation))
    cf_vali%>%
      map_dfr(~ cbind(as.data.frame(t(.x$overall)),as.data.frame(t(.x$byClass))), .id = "Modelname")%>%
      select(Modelname, Accuracy, Kappa)
    
    accuarcy_all_vali <- predict(mg_final_tra()$model_fits, newdata = x_validation)%>% 
      map( ~confusionMatrix(.x, y_validation))%>%
      map_dfr(~ cbind(as.data.frame(t(.x$overall)),as.data.frame(t(.x$byClass))), .id = "Modelname")%>%
      select(Modelname, Accuracy, Kappa)
    accuracy_auc_vali <-  bind_cols(accuarcy_all_vali,auc_value_df_vali_df)
    print(accuracy_auc_vali)
  })
  
  output$best_model_vali <- renderUI({
    
    AUC2 <- mg_final_tra()$model_fits%>% 
      predict(., newdata = x_validation, type ="prob")%>% 
      map(~roc(y_validation, .x[,2]))
    
    auc_value_vali <- map_dbl(AUC2, ~(.x)$auc)
    auc_value_df_vali_df <- as.data.frame(auc_value_vali)
    
    cf_vali <- predict(mg_final_tra()$model_fits, newdata = x_validation)%>%
      map( ~confusionMatrix(.x, y_validation))
    cf_vali%>%
      map_dfr(~ cbind(as.data.frame(t(.x$overall)),as.data.frame(t(.x$byClass))), .id = "Modelname")%>%
      select(Modelname, Accuracy, Kappa)
    
    accuarcy_all_vali <- predict(mg_final_tra()$model_fits, newdata = x_validation)%>% 
      map( ~confusionMatrix(.x, y_validation))%>%
      map_dfr(~ cbind(as.data.frame(t(.x$overall)),as.data.frame(t(.x$byClass))), .id = "Modelname")%>%
      select(Modelname, Accuracy, Kappa)
    
    accuracy_auc_vali <-  bind_cols(accuarcy_all_vali,auc_value_df_vali_df) 
    
    max_auc_vali <- filter(accuracy_auc_vali, auc_value_vali == max(accuracy_auc_vali$auc_value_vali))%>%
      select(Modelname)
    max_Accuracy_vali <- filter(accuracy_auc_vali, Accuracy == max(accuracy_auc_vali$Accuracy))%>%
      select(Modelname)
    
    
    HTML(paste("Results", br(),  
               "1- The model", strong(max_auc_vali), "has the highest AUC value.", br(),
               "2- The model",strong(max_Accuracy_vali), "has the highest Accuracy."))
  })
  
  ## confusion matrix on data validation
  output$accuracy_vali<- renderPrint({if(input$abd == "none"){print("Please select a model")}
    else{
      confusionMatrix(predict(mg_final_tra()$model_fits[[input$abd]], x_validation), y_validation)}
  })
  
  ############################## Model explaination  #######################################
  ############## Using IML Package ###################################
  ## define the IML predictor
  mod2 <- reactive({ Predictor$new(mg_final_tra()$model_fits[[input$perf]], 
                                   data = x_validation, y= y_validation,  type = "prob")})
  
  ## plot the variable importance
  output$condition <- renderPrint({
    if(input$perf == "none"){print("Please select a model")}
    else{ 
      
      output$variable_imp <- renderPlot({
        vari_imp <- FeatureImp$new(mod2(), loss = "ce")
        plot(vari_imp) + theme_minimal(base_size = 24)
        })
    }})
  ## Visualise the features interaction 
  output$inter_action <- renderPlot({
    plot(Interaction$new(mod2())) + theme_minimal(base_size = 24)
  })
  
  output$condition2 <- renderPrint({
    if(input$Variab_le == "none"){print("Please select a variable")}
    else{
      output$two_way_inter <- renderPlot({
        plot(Interaction$new(mod2(), feature = input$Variab_le)) +
          theme_minimal(base_size = 24)
        
        })}
  })
  
  ## Feature effects 
  output$PD_P <-  renderPlot({
    ale <- FeatureEffect$new(mod2(), feature = input$Variab_le)
    plot(ale) + 
      theme_minimal(base_size = 24) +
      ggtitle("Partial Dependence Plot")
  })
  
  output$des <- renderUI({
    br()
    HTML(paste(
      strong("Accumulated local effects describe how features influence the prediction 
        of a machine learning model on average. ALE plots 
        are a faster and unbiased alternative to partial dependence Plots (PDPs).")))
  })
  
  ## Plot PDP and ICE
  output$IC_E <- renderPlot({ 
    plot(FeatureEffect$new(mod2(), feature = input$Variab_le, method = "pdp+ice")) + theme_minimal(base_size = 24)
  })
  
  ## Surrogate model
  output$surrogate_model <- renderPlot({
    tree <-  TreeSurrogate$new(mod2(), maxdepth = 2)
    plot(tree) + theme_minimal(base_size = 24)
  })
  
  ## Shapley Values
  # Explain single predictions with game theory
  
  output$shapley_Values <- renderPlot({
    shapley <- reactive ({Shapley$new(mod2(), x.interest = x_validation[input$row_index, ])})
    plot(shapley()) + theme_minimal(base_size = 24)
    
  })  
  
  ## LIME
  # Explain single prediction with LIME 
  
  output$single_plot_lime <- renderPlot({
    
    lime.explain = LocalModel$new(mod2(),x.interest = x_validation[input$row_index,])
    lime.explain$results %>%
      ggplot(aes(x = reorder(feature.value, -effect), y = effect, fill = .class)) +
      facet_wrap(~ .class, ncol = 1) +
      geom_bar(stat = "identity", alpha = 1) +
      scale_fill_discrete() +
      coord_flip() +
      labs(title = paste0("Instance row number: ", input$row_index)) +
      guides(fill = FALSE) +
      theme_minimal(base_size = 24)
  })
  
  ############## using DALEX Package  ###################################
  ## definde DALEX explainer 
  y_validation_da <- factor(ifelse(y_validation == "Yes", 1, 0))
  yTest = as.numeric(as.character(y_validation_da))
  p_fun <- function(object, newdata){predict(object, newdata= newdata, type="prob")[,2]}
  
  explainer_da <- reactive({DALEX::explain(mg_final_tra()$model_fits[[input$perf]], label = input$perf,
                                     data = x_validation, y= yTest,
                                     predict_function = p_fun)
    
  })
  ## plot  variable importance with DALEX
  output$condition_da <- renderPrint({
    if(input$perf == "none"){print("Please select a model")}
    else{ 
      output$variable_imp_da <- renderPlot({
        vari_imp_da <-  feature_importance(explainer_da(), loss_function = loss_root_mean_square)
        plot(vari_imp_da) + theme_minimal(base_size = 24)
          
      })
    }})
  
  ## PDP plot
  output$Vari_resp <- renderPlot({
    Vari_response <- ingredients::partial_dependency( explainer_da(), variables = input$Variab_le)
    plot(Vari_response) + theme_minimal(base_size = 24)
  })
  ## ALE plot
  output$ALE_da <- renderPlot({
    accumulated_da <- accumulated_dependency( explainer_da(), variables = input$Variab_le)
    plot( accumulated_da) + theme_minimal(base_size = 24)
  })
  
  ## What if Profiles/ICE
  output$what_if <- renderPlot({
    cp_model <- ceteris_paribus(explainer_da(), x_validation[input$row_index, ])
    plot(cp_model, variables = input$Variab_le) +
      show_observations(cp_model, variables = input$Variab_le) +
      theme_minimal(base_size = 24) +
      ggtitle("Ceteris Paribus Profiles of the selected model")
  })
  ## BreakDown Plot
  output$single_plot_brd_da <- renderPlot({
    brd_da <- break_down(explainer_da(), 
                         x_validation[input$row_index, ], keep_distributions = TRUE)
    plot(brd_da) + theme_minimal(base_size = 24)
  })
  ## Description
  output$descri_ption <- renderUI({ 
    brd_da <- break_down(explainer_da(), x_validation[input$row_index, ], 
                         keep_distributions = TRUE)
    br()
    HTML(paste(strong("Description:"),  
               br(), 
               describe(brd_da,
                        short_description = FALSE,
                        display_values =  TRUE,
                        display_numbers = TRUE,
                        display_distribution_details = FALSE)))
  })
  
  ## Shapley Values 
  # Explain single predictions with game theory
  output$shapley_Values_da <- renderPlot({
    shap_values <- shap(explainer_da(), x_validation[input$row_index, ], B=25)
    plot(shap_values)  + theme_minimal(base_size = 24)
  })
  
  output$descri_ption_sharp <- renderUI({
    shap_values <- shap(explainer_da(), x_validation[input$row_index, ], B=25)
    br()  
    HTML(paste(
      strong("Description:"),  
      br(), 
      describe(shap_values,
               short_description = FALSE,
               display_values =  TRUE,
               display_numbers = TRUE,
               display_distribution_details = FALSE)))
  })
  
  ## Generates a website with HTML summaries for the selected predictive model 
  observeEvent(input$do, 
               {modelDown::modelDown( explainer_da())})
  
  output$down_mod <- renderUI({HTML(paste("to summary: auditor, drifter, model_performance, 
                                                variable_importance, and variable_response"))})
  
  ################################# Run the application ######################### 
}
shinyApp(ui = ui, server = server)

