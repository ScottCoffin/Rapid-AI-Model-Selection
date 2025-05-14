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
library(partykit)
library(DT)

####################
# load data
set.seed(9650)

# Load the penguins dataset and prepare it
data("penguins")
# Prepare the dataset
penguins <- penguins %>%
  dplyr::rename("response" = "sex") %>%
  drop_na(response) %>%
  # Introduce NAs with a 10% chance for demonstration
  dplyr::mutate(dplyr::across(c(where(is.numeric), -response), 
                              ~ ifelse(runif(length(.)) < 0.1, NA, .)))  # 10% chance to replace with NA

################## USER INTERFACE ##################
ui <- fluidPage(theme = shinytheme("yeti"),
                tags$head(tags$link(rel = "stylesheet", type = "text/css", href = "styles.css")),
                ## Application title
                titlePanel(wellPanel("Rapid AI Builder App")),
                tags$div("Contact: Scott.Coffin@oehha.ca.gov", align = 'center', style = 'font-size: 15px; display: block; margin-left: auto; margin-right: auto;'), 
                navbarPage("Workflow ===>",
                           tabPanel("Exploratory Data Analysis",
                                    tabsetPanel(type = "tabs",
                                                tabPanel("Upload data",
                                                         div(style = "width: 100%",
                                                         fileInput("csvFile", "Upload a CSV or MS Excel file with any number of columns and rows (however ideally having fewer columns than rows). Ensure a columns is named `response`. If using Excel - only the first sheet will be used)",
                                                                   accept = c(
                                                                     ".xlsx",
                                                                     "text/csv",
                                                                     "text/comma-separated-values,text/plain",
                                                                     ".csv")),
                                                         ),
                                                         downloadButton("example_download", "Download Example Dataset (Penguins)",
                                                                        style = "color: #fff; background-color: #0097ab; border-color: #2e6da4; font-size: 15px; padding: 10px 15px;")
                                                         ),
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
                                                              choices = c("none"),
                                                              selected = "none", 
                                                              width = '200px'))),
                                      mainPanel(
                                        tabsetPanel(type= "tabs",
                                                    # tabPanel("Choose Models to Train",
                                                    #          navlistPanel(
                                                    #            tabPanel("Model Choices"),
                                                    #            actionButton("trainBtn", "Train Models",
                                                    #                         style = "color: #fff; background-color: #077336; border-color: #2e6da4; font-size: 40px; padding: 10px 15px;"),
                                                    #          )
                                                    #          ),
                                                    ## output model training and summary
                                                    tabPanel("Model Training & Summary",
                                                             navlistPanel(
                                                               tabPanel("1- Show info model grid ",verbatimTextOutput("info_modelgrid")),
                                                               tabPanel("2- Performance statistics of the model grid (dotplot) ", 
                                                                        p("Note that this step can take between 5 and 60+ minutes depending on the size of your dataset. Consider running this app locally to speed up processing."),
                                                                        withSpinner(plotOutput("dotplot", width = 600, height = 600))),
                                                               tabPanel("3- Extract Performance of the model grid ", 
                                                                        #withSpinner(verbatimTextOutput(outputId = "summary"))
                                                                        withSpinner(DTOutput(outputId = "summary"))
                                                                        ),
                                                               tabPanel("4- Show the AUC & Accuracy of individual models (on data training)",
                                                                        withSpinner(DTOutput("Accurac_AUC")), 
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
                                                                        DTOutput(outputId = "AUC_of_individ"),
                                                                        htmlOutput("best_model_vali")),
                                                               tabPanel("2- Show single model's Accuracy/data validation",
                                                                        verbatimTextOutput("accuracy_vali"))))
                                                    
                                        )))),
                           ###########################################################  Module post hoch explanation ################  
                           tabPanel("Model Explanation",
                                    sidebarLayout(
                                      sidebarPanel(
                                        wellPanel(selectInput(inputId = "perf", 
                                                              label = "Choose Model: ",
                                                              choices = c("none"),
                                                              selected = "none", width = '200px')),
                                        
                                        wellPanel(selectInput(inputId = "Variab_le", 
                                                              label = "Choose Variable: ",
                                                              choices = c("none"),
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
                                                                                    tabPanel("1- Plot variable importance", 
                                                                                            withSpinner(plotOutput("variable_imp", height = "600px")), 
                                                                                            verbatimTextOutput("condition")
                                                                                            ),
                                                                                    tabPanel("2- Effect of a feature on predictions", 
                                                                                             withSpinner(plotOutput("PD_P", height = "600px")), 
                                                                                             htmlOutput("des")
                                                                                                         ),
                                                                                    tabPanel("3- Feature Interaction",
                                                                                             withSpinner(plotOutput("inter_action", height = "600px"))
                                                                                                         ),
                                                                                    tabPanel("4- 2-way interactions", 
                                                                                             withSpinner(plotOutput("two_way_inter", height = "600px")), 
                                                                                             verbatimTextOutput("condition2")),
                                                                                    tabPanel("5- ICE + PDP plots", 
                                                                                             withSpinner(plotOutput("IC_E"))
                                                                                                         ),
                                                                                    tabPanel("6- Global Surrogate model", 
                                                                                             withSpinner(plotOutput("surrogate_model", height = "600px")))
                                                                                    )),
                                                                         tabPanel("Local Interpretation",
                                                                                  navlistPanel(
                                                                                    tabPanel("1- Shapley Values", 
                                                                                             withSpinner(plotOutput(outputId = "shapley_Values", height = "600px"))
                                                                                                         ),
                                                                                    tabPanel("2- LIME", 
                                                                                             withSpinner(plotOutput(outputId = "single_plot_lime", height = "600px"))
                                                                                             ))))),
                                                    #### DALEX Packge#######          
                                                    tabPanel("DALEX Package",
                                                             tabsetPanel(type = "tabs",
                                                                         tabPanel("Global Interpretation ",
                                                                                  navlistPanel(
                                                                                    tabPanel("1- Plot variable importance", 
                                                                                             withSpinner(plotOutput("variable_imp_da", height = "600px")), 
                                                                                             verbatimTextOutput("condition_da")),
                                                                                    tabPanel("2- Variable response", 
                                                                                             plotOutput("Vari_resp", height = "600px")),
                                                                                    tabPanel("3- Accumulated Local Effect (ALE)",
                                                                                             plotOutput("ALE_da"), 
                                                                                             verbatimTextOutput("condition2_da")),
                                                                                    tabPanel("4- What-If Profiles/ICE analysis ", 
                                                                                             plotOutput("what_if", height = "600px")))),
                                                                         
                                                                         tabPanel("Local Interpretation ",
                                                                                  navlistPanel(
                                                                                    tabPanel("1- Shapley Values", 
                                                                                             withSpinner(plotOutput(outputId = "shapley_Values_da", height = "600px")), 
                                                                                             htmlOutput("descri_ption_sharp")),
                                                                                    tabPanel("2- BreakDown", 
                                                                                             withSpinner(plotOutput(outputId = "single_plot_brd_da", 
                                                                                                        height = "600px")),
                                                                                             htmlOutput("descri_ption")))),
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
  
  ## Reactive expression to store the uploaded data
  data_set <- reactive({
    req(input$csvFile)  # Ensure the file input is available
    inFile <- input$csvFile
    
    # Determine file type based on extension
    file_ext <- tools::file_ext(inFile$name)
    
    if (file_ext == "csv") {
      data <- read.csv(inFile$datapath, stringsAsFactors = FALSE)
    } else if (file_ext == "xlsx") {
      data <- readxl::read_excel(inFile$datapath, sheet = 1)
    } else {
      showNotification("Unsupported file type. Please upload a CSV or Excel file.", type = "error")
      return(NULL)  # Exit if the file type is unsupported
    }
    
    # Check if the response variable exists
    if (!("response" %in% names(data))) {
      showNotification("No 'response' column found in the uploaded file.", type = "error")
      return(NULL)  # Exit if the 'response' column is not present
    }
    
    return(data)  # Return the data if everything is fine
  })
  
 
  # Download handler for the penguins dataset
  output$example_download <- downloadHandler(
    filename = function() {
      paste("penguins_dataset.csv")  # Filename 
    },
    content = function(file) {
      write.csv(penguins, file, row.names = FALSE)  # Write the dataset to a CSV file
    }
  )
  
  ## Reactive preprocessing using recipes
  processed_data <- reactive({
    req(data_set())  # Ensure the data is available
    data_set <- data_set()
    
    set.seed(9650)
    Attri_rec_2 <- 
      recipe(response ~ ., data = data_set) %>%        # Formula.
      step_unknown(all_nominal_predictors()) %>% 
      step_dummy(all_nominal_predictors()) %>%               # Convert nominal data into one or more numeric.
      step_impute_knn(all_predictors()) %>%           # Impute missing data using nearest neighbors.
      step_zv(all_predictors()) %>%                 # Remove variables that are highly sparse and unbalanced.
      step_corr(all_predictors()) %>%               # Remove variables that have large absolute correlations with other variables.
      step_center(all_predictors()) %>%             # Normalize numeric data to have a mean of zero.
      step_scale(all_predictors()) %>%              # Normalize numeric data to have a standard deviation of one.
      prep(training = data_set, retain = TRUE)    # Train the data recipe 
    
    processed_data <- as.data.frame(juice(Attri_rec_2))
    processed_data$response <- data_set$response  # Add the response variable
    return(processed_data)
  })
  
  ## Reactive training and validation datasets
  train_validation_data <- reactive({
    req(processed_data())  # Ensure the processed data is available
    processed_data <- processed_data()
    
    set.seed(9650)
    validation_index <- createDataPartition(processed_data$response, times = 1, p = 0.70, list = FALSE)
    
    # Split into training and validation datasets
    data_train <- processed_data[validation_index, ]
    data_validation <- processed_data[-validation_index, ]
    
    train_validation_data <-  list(
      train = data_train,
      validation = data_validation,
      x_train = data_train %>% dplyr::select(-response),
      y_train = data_train[["response"]],
      x_validation = data_validation %>% dplyr::select(-response),
      y_validation = data_validation[["response"]]
    )
    return(train_validation_data)
  })
  
  ### Vaiable list
   variable_list <- reactive({
     req(train_validation_data())  # Ensure training data is available
     train_validation_data <- train_validation_data()
     variable_list <- colnames(train_validation_data$x_validation)
     return(variable_list)
   }) 
   
   # Update the selectInput choices based on the reactive list_model
   observe({
     updateSelectInput(session, "Variab_le", 
                       choices = c("none", variable_list()),  # Include "none" and the reactive models
                       selected = "none")  # Reset to "none" if needed
   })
  
  ## Reactive model grid
  mg_final <- reactive({
    req(train_validation_data())  # Ensure training data is available
    train_validation_data <- train_validation_data()
    
    set.seed(9650)
    mg <-
      model_grid() %>%
      share_settings(
        y = train_validation_data$y_train,
        x = train_validation_data$x_train,
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
    
    return(mg_final)
  })
  
  list_model <- reactive({
    mg_final <- mg_final()
    list_model <- c(names(mg_final$models))
    return(list_model)
  }) 
  
  # Update the selectInput choices based on the reactive list_model
  observe({
    updateSelectInput(session, "perf", 
                      choices = c("none", list_model()),  # Include "none" and the reactive models
                      selected = "none")  # Reset to "none" if needed
  })
  
  # Update the selectInput choices based on the reactive list_model
  observe({
    updateSelectInput(session, "abd", 
                      choices = c("none", list_model()),  # Include "none" and the reactive models
                      selected = "none")  # Reset to "none" if needed
  })
  
  
  ## Outputs
  output$dataTable <- renderTable({
    req(data_set())  # Ensure the reactive data is available
    head(data_set())  # Display the first few rows of the uploaded data
  })
  
  output$str_ucture <- renderPrint({
    req(data_set())
    str(data_set())
  })
  
  output$missing_value <- renderPlot({
    req(data_set())
    data_set <- data_set()
    plot_missing(data_set, ggtheme = theme_minimal(base_size = 24))
  })
  
  output$coor_plot <- renderPlot({
    req(data_set())
    plot_correlation(data_set(), ggtheme = theme_minimal(base_size = 24))
  })
  
  ## after processing the data 
  # data structure
  output$str_ucture_after <- renderPrint({
    req(data_set())
    processed_data <- processed_data()
    str(processed_data)
  })
  # plot to check the missing values
  output$missing_value_after <- renderPlot({
    req(data_set())
    processed_data <- processed_data()
    plot_missing(processed_data,
                 ggtheme = theme_minimal(base_size = 24))
  })
  # plot to check the correlation
  output$coor_plot_after <- renderPlot({
    req(data_set())
    processed_data <- processed_data()
    plot_correlation(processed_data,
                     ggtheme = theme_minimal(base_size = 24))
  })
  set.seed(9650)
  ## Infos about the componets of the constructed model grid     
  output$info_modelgrid <- renderPrint({
    req(mg_final())
    mg_final <- mg_final()
    mg_final$models
  })
  ## train all models 
  set.seed(9650)
  #mg_final_tra <- eventReactive(input$trainBtn, {
  mg_final_tra <- reactive({
    req(mg_final())
    mg_final <- mg_final()
    train_validation_data <- train_validation_data()
    list_model <- list_model()
    
    mg_final_tra <- caret::train(mg_final)
    return(mg_final_tra)
    })
  
  ## plot to compare
  output$dotplot <- renderPlot({ 
    
    if (is.null(mg_final_tra()$model_fits)) {
      return("No models have been trained.")
    }
    
    mg_final_tra <- mg_final_tra()
    train_validation_data <- train_validation_data()
    x_train <- train_validation_data$x_train
    y_train <- train_validation_data$y_train
    
    mg_final_tra$model_fits %>%
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
  output$summary <- renderDT({
    mg_final_tra <- mg_final_tra()
    
    # mg_final_tra$model_fits %>%
    #   caret::resamples(.) %>%
    #   summary(.)
    
    # Get resampling results
    resamples_results <- caret::resamples(mg_final_tra$model_fits)
    
    # Create a summary of the resampling results
    summary_results <- summary(resamples_results)
    
    # Convert the summary into a data frame for display
    summary_df <- as.data.frame(summary_results$statistics) %>% 
      dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ signif(., 4))) %>% 
      t()
    
    # Render the data frame as a pretty table
    DT::datatable(summary_df, options = list(pageLength = 10, autoWidth = TRUE))
  })
  
  ## computing the auc & the accuracy (on train data)
  set.seed(9650)
  
  accuracy_auc_train <- reactive({
    req(mg_final_tra())
    mg_final_tra <- mg_final_tra()
    req(train_validation_data())
    train_validation_data <- train_validation_data()
    x_train <- train_validation_data$x_train
    y_train <- train_validation_data$y_train
    
    # Ensure y_train is a factor
    y_train <- as.factor(y_train)
    
    # Calculate AUC for each model
    AUC1 <- purrr::map(mg_final_tra$model_fits, ~ {
      # Predict probabilities for the current model
      pred <- predict(.x, newdata = x_train, type = "prob")
      
      # Ensure the prediction has the correct structure
      if (!is.data.frame(pred) || ncol(pred) < 2) {
        stop("Error: Predictions must be a data frame with at least two columns.")
      }
      
      # Calculate ROC for the second column (assumed to be the positive class)
      pROC::roc(y_train, pred[, 2])
    })
    
    # Extract AUC values
    auc_value_train <- purrr::map_dbl(AUC1, ~ .x$auc)
    auc_value_df_train_df <- as.data.frame(auc_value_train)
    
    # Calculate accuracy and kappa for each model
    accuarcy_all_train <- purrr::map(mg_final_tra$model_fits, ~ {
      # Predict class labels for the current model
      pred <- predict(.x, newdata = x_train)
      
      # Ensure predictions are factors with the same levels as y_train
      pred <- factor(pred, levels = levels(y_train))
      
      # Compute confusion matrix
      cm <- caret::confusionMatrix(pred, y_train)
      
      # Combine overall and by-class metrics into a single data frame
      cbind(as.data.frame(t(cm$overall)), as.data.frame(t(cm$byClass)))
    }) %>%
      purrr::map_dfr(~ ., .id = "Modelname")  # Combine results into a single data frame
    
    # Ensure the resulting data frame has the expected columns
    if (!all(c("Modelname", "Accuracy", "Kappa") %in% colnames(accuarcy_all_train))) {
      stop("Error: The resulting data frame does not contain the expected columns.")
    }
    
    # Select the desired columns
    accuarcy_all_train <- accuarcy_all_train %>%
      dplyr::select(Modelname, Accuracy, Kappa)
    
    # Combine accuracy and AUC into one data frame
    accuracy_auc_train <- bind_cols(accuarcy_all_train, auc_value_df_train_df) %>%
      # Format numeric columns to 4 significant digits
      dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ signif(., 4)))
    
    # Remove row names
    rownames(accuracy_auc_train) <- NULL
    
    accuracy_auc_train
  })
    
    output$Accurac_AUC <- renderDT({
    # Render the data table
    DT::datatable(accuracy_auc_train())
  })
  
  
  
  ## computing the auc and the Accuracy
  set.seed(9650)
  output$best_model_train <- renderUI({
    accuracy_auc_train  <- accuracy_auc_train()
    # Select the desired columns
    accuarcy_all_train <- accuarcy_all_train %>%
      dplyr::select(Modelname, Accuracy, Kappa)
    
    max_auc_train <- filter(accuracy_auc_train, auc_value_train == max(accuracy_auc_train$auc_value_train))%>%
      dplyr::select(Modelname)
    
    max_Accuracy_train <- filter(accuracy_auc_train, Accuracy == max(accuracy_auc_train$Accuracy))%>%
      dplyr::select(Modelname)
    
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
  output$accuracy <- renderPrint({
    if (input$abd == "none") {
      print("Please select a model")
    } else { 
      req(mg_final_tra())
      mg_final_tra <- mg_final_tra()
      req(train_validation_data())
      train_validation_data <- train_validation_data()
      x_train <- train_validation_data$x_train
      y_train <- train_validation_data$y_train
      
      # Ensure y_train is a factor
      y_train <- as.factor(y_train)
      
      # Predict class labels for the selected model
      pred <- predict(mg_final_tra$model_fits[[input$abd]], x_train)
      
      # Ensure predictions are factors with the same levels as y_train
      pred <- factor(pred, levels = levels(y_train))
      
      # Compute and return the confusion matrix
      caret::confusionMatrix(pred, y_train)
    }
  })
  
  
  ########################### Model Validation #########################
  # Extract the auc values
  accuracy_auc_vali <- reactive({
    req(train_validation_data())
    train_validation_data <- train_validation_data()
    req(mg_final_tra())
    mg_final_tra <- mg_final_tra()
    # Extract training and validation data
    x_train <- train_validation_data$x_train
    y_train <- as.factor(train_validation_data$y_train)
    x_validation <- train_validation_data$validation
    y_validation <- as.factor(train_validation_data$validation$response)
    
    # Calculate AUC for each model
    AUC2 <- purrr::map(mg_final_tra$model_fits, ~ {
      # Predict probabilities for the current model
      prob <- predict(.x, newdata = x_validation, type = "prob")
      
      # Check if the prediction is a data frame with the expected columns
      if (!is.data.frame(prob) || ncol(prob) < 2) {
        stop("Error: Predictions must be a data frame with at least two columns.")
      }
      
      # Calculate ROC for the "male" column (assumed to be the positive class)
      pROC::roc(y_validation, prob$male)  # Use the column name directly
    })
    
    auc_value_vali <- map_dbl(AUC2, ~(.x)$auc)
    auc_value_df_vali_df <- as.data.frame(auc_value_vali)
    
    # Calculate confusion matrices for each model
    cf_vali <- purrr::map(mg_final_tra$model_fits, ~ {
      # Predict class labels for the current model
      pred <- predict(.x, newdata = x_validation, type = "raw")
      
      # Ensure predictions are factors with the same levels as y_validation
      pred <- factor(pred, levels = levels(y_validation))
      
      # Compute the confusion matrix
      caret::confusionMatrix(pred, y_validation)
    })
    
    cf_vali%>%
      map_dfr(~ cbind(as.data.frame(t(.x$overall)),as.data.frame(t(.x$byClass))), .id = "Modelname")%>%
      dplyr::select(Modelname, Accuracy, Kappa)
    
    
    # Calculate accuracy and kappa for each model
    accuarcy_all_vali <- purrr::map(mg_final_tra$model_fits, ~ {
      # Predict class labels for the current model
      pred <- predict(.x, newdata = x_validation)
      
      # Ensure predictions are factors with the same levels as y_train
      pred <- factor(pred, levels = levels(y_validation))
      
      # Compute confusion matrix
      cm <- caret::confusionMatrix(pred, y_validation)
      
      # Combine overall and by-class metrics into a single data frame
      cbind(as.data.frame(t(cm$overall)), as.data.frame(t(cm$byClass)))
    }) %>%
      purrr::map_dfr(~ ., .id = "Modelname")  # Combine results into a single data frame
    
    # Ensure the resulting data frame has the expected columns
    if (!all(c("Modelname", "Accuracy", "Kappa") %in% colnames(accuarcy_all_vali))) {
      stop("Error: The resulting data frame does not contain the expected columns.")
    }
    
    # Select the desired columns
    accuarcy_all_vali <- accuarcy_all_vali %>%
      dplyr::select(Modelname, Accuracy, Kappa)
    
    accuracy_auc_vali <-  bind_cols(accuarcy_all_vali,auc_value_df_vali_df) %>% 
      dplyr::mutate(dplyr::across(dplyr::where(is.numeric), ~ signif(., 4)))
    
    return(accuracy_auc_vali)
  })
  
  output$AUC_of_individ <- renderDT({
    DT::datatable(accuracy_auc_vali())
  })
  
  output$best_model_vali <- renderUI({
    accuracy_auc_vali <- accuracy_auc_vali()
    
    max_auc_vali <- filter(accuracy_auc_vali,
                           auc_value_vali == max(accuracy_auc_vali$auc_value_vali)) %>%
      dplyr::select(Modelname)
    
    max_Accuracy_vali <- filter(accuracy_auc_vali, 
                                Accuracy == max(accuracy_auc_vali$Accuracy)) %>%
      dplyr::select(Modelname)
    
    
    HTML(paste("Results", br(),  
               "1- The model", strong(max_auc_vali), "has the highest AUC value.", br(),
               "2- The model",strong(max_Accuracy_vali), "has the highest Accuracy."))
  })
  
  ## confusion matrix on data validation
  output$accuracy_vali <- renderPrint({
    if (input$abd == "none") {
      print("Please select a model")
    } else {
      req(train_validation_data())
      train_validation_data <- train_validation_data()
      req(mg_final_tra())
      mg_final_tra <- mg_final_tra()
      
      # Extract training and validation data
      x_train <- train_validation_data$x_train
      y_train <- as.factor(train_validation_data$y_train)
      x_validation <- train_validation_data$validation
      y_validation <- as.factor(train_validation_data$validation$response)
      
      confusionMatrix(predict(mg_final_tra$model_fits[[input$abd]], train_validation_data$x_validation), y_validation)
    }
  })
  
  
  ############################## Model explaination  #######################################
  ############## Using IML Package ###################################
  ## define the IML predictor
  mod2 <- reactive({
    req(mg_final_tra())
    train_validation_data <- train_validation_data()
    Predictor$new(mg_final_tra()$model_fits[[input$perf]], 
                  data = train_validation_data$x_validation, y = train_validation_data$y_validation,  type = "prob")
    })
  
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
    mg_final_tra <- mg_final_tra()
    req(train_validation_data())
    train_validation_data <- train_validation_data()
    x_train <- train_validation_data$x_train
    y_train <- train_validation_data$y_train
    x_validation <- train_validation_data$x_validation
    
    shapley <- reactive ({Shapley$new(mod2(), x.interest = x_validation[input$row_index, ])})
    plot(shapley()) + theme_minimal(base_size = 24)
    
  })  
  
  ## LIME
  # Explain single prediction with LIME 
  
  output$single_plot_lime <- renderPlot({
    mg_final_tra <- mg_final_tra()
    req(train_validation_data())
    train_validation_data <- train_validation_data()
    x_train <- train_validation_data$x_train
    y_train <- train_validation_data$y_train
    x_validation <- train_validation_data$x_validation
    
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
  # Reactive y_validation_da and yTest
  y_validation_da <- reactive({
    req(train_validation_data())  # Ensure train_validation_data is available
    factor(ifelse(train_validation_data()$y_validation == "Yes", 1, 0))
  })
  
  yTest <- reactive({
    as.numeric(as.character(y_validation_da()))
  })
  
  # Predict function
  p_fun <- function(object, newdata) {
    predict(object, newdata = newdata, type = "prob")[, 2]
  }
  
  explainer_da <- reactive({
    req(mg_final_tra())
    train_validation_data <- train_validation_data()
    
    DALEX::explain(mg_final_tra()$model_fits[[input$perf]], 
                   label = input$perf,
                   data = train_validation_data$x_validation, 
                   y = yTest(),
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
    req(mg_final_tra())
    train_validation_data <- train_validation_data()
    x_validation <- train_validation_data$x_validation
    
    cp_model <- ceteris_paribus(explainer_da(), x_validation[input$row_index, ])
    plot(cp_model, variables = input$Variab_le) +
      show_observations(cp_model, variables = input$Variab_le) +
      theme_minimal(base_size = 24) +
      ggtitle("Ceteris Paribus Profiles of the selected model")
  })
  ## BreakDown Plot
  output$single_plot_brd_da <- renderPlot({
    mg_final_tra <- mg_final_tra()
    req(train_validation_data())
    train_validation_data <- train_validation_data()
    x_train <- train_validation_data$x_train
    y_train <- train_validation_data$y_train
    x_validation <- train_validation_data$x_validation
    
    brd_da <- break_down(explainer_da(), 
                         x_validation[input$row_index, ], keep_distributions = TRUE)
    plot(brd_da) + theme_minimal(base_size = 24)
  })
  ## Description
  output$descri_ption <- renderUI({ 
    req(mg_final_tra())
    train_validation_data <- train_validation_data()
    x_validation <- train_validation_data$x_validation
    
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
    req(mg_final_tra())
    train_validation_data <- train_validation_data()
    x_validation <- train_validation_data$x_validation
    
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

