import ConfigParser
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import logging

# Constants
CONTROL = "Control"
TEST = "Test"
P_VALUE_PRECISION = 3
ALPHA_VALUE = 0.05

# prepare configuration from config file
def prepare_conf() :
    config_parser = ConfigParser.RawConfigParser()
    conf_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Config.txt')
    config_parser.read(conf_file)
    return config_parser


# prepare logger
def prepare_logger() :
    logger = logging.getLogger('prodtest')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger


config_parser = prepare_conf()
logger = prepare_logger()


# Main
def main():
    input = config_parser.get('data', 'input_file_path')
    logger.info("Reading input file from path: {}".format(input))
    csv_file = pd.read_csv(input, skip_blank_lines=True)

    logger.info("Parsing control and test group data by adserver_host")
    test_servers = config_parser.get('data', 'test_servers').split(",")
    control_group = csv_file.loc[csv_file['adserver_host'].isin(test_servers) == False]
    test_group = csv_file.loc[csv_file['adserver_host'].isin(test_servers)]

    logger.info("Setting up plot configuration")
    plot_config()

    output_type = config_parser.get('data', 'output_type')
    logger.info("Generating {} report".format(output_type))
    generate_report(csv_file, control_group, test_group, output_type)


# Create design configuration for seaborn plot
def plot_config():
    sns.set_style(config_parser.get('design', 'seaborn_style'))
    sns.set_palette(config_parser.get('design', 'seaborn_palette'))


# Check if the data is distributed normally
def is_normal_distribution(data):
    norm_res, norm_pval = stats.shapiro(data)
    is_normal_distribution = norm_pval > ALPHA_VALUE;
    return is_normal_distribution


# Run a stats test according to the distribution of the data
def do_test(control_data, test_data, is_normal_distribution):
    report_text = "Distribution Type: {}\nTest Performed: {}\nTest Summary: {}\n\nTest Details\nControl Group: {}\n\nTest Group: {}\n"
    if (is_normal_distribution):
        dist_type = "Normal Distribution"
        test_type = "T-Test"
        test_res, test_pval = stats.ttest_ind(control_data, test_data, axis=0, equal_var=False)
    else:
        dist_type = "Unknown Distribution"
        test_type = "Mann-Whitney U Test"
        test_res, test_pval = stats.mannwhitneyu(control_data, test_data)

    is_significant_diff = test_pval < ALPHA_VALUE
    if (is_significant_diff):
        test_summary = "The difference is significant with p-value {}".format(test_pval)
    else:
        test_summary = "All Good!"

    report_text = report_text.format(dist_type, test_type, test_summary, control_data.describe(), test_data.describe())
    return report_text


# Create a distribution plot from data
def create_distribution_plot(pdf, column_name, control_data, test_data):
    sns.distplot(control_data, label=CONTROL)
    sns.distplot(test_data, label=TEST)
    plt.legend()  # Show lables
    plt.title("Distribution for ".format(column_name))
    save_curr_figure_and_close(pdf)


# Create a plot with the behaviour or each instance
def create_instance_behaviour_plot(pdf, column_name, group_name, data):
    #time = pd.to_datetime(data['tslice'])
    adserver_data = pd.DataFrame(data=data).groupby(['tslice', 'adserver_host'])
    adserver_data.agg(column_name).sum().unstack().plot()
    plt.title("{} Per Test Instance ({})".format(column_name, group_name))
    plt.xlabel("Time")
    plt.ylabel(column_name)
    save_curr_figure_and_close(pdf)

# Create a box plot from data
def create_boxplot(pdf, column_name, control_data, test_data):
    boxplot_data = pd.DataFrame({
        CONTROL : control_data,
        TEST : test_data
    })
    sns.boxplot(data=boxplot_data, orient="v")
    plt.xlabel("Experiment Groups")
    plt.ylabel(column_name)
    save_curr_figure_and_close(pdf)


# Save and close current pdf figure
def save_curr_figure_and_close(pdf):
    pdf.savefig(plt.gcf())
    plt.close()


# Add report text to current pdf figure
def add_report_text(pdf, column_name, text):
    plt.gcf().text(0.05, 0.85, "Evaluating {} \n".format(column_name), transform=plt.gcf().transFigure, size=18)
    plt.gcf().text(0.05, 0.1, text, transform=plt.gcf().transFigure)
    save_curr_figure_and_close(pdf)


# Add report summary to first page
def add_report_summary(output_file) :
    report_file = PdfFileReader(output_file, 'rb')
    report_file.addPage()


# Generate the A/B test report in PDF format
def generate_pdf_report(csv_file, control_group, test_group):
    prodtest_name = config_parser.get('data', 'prodtest_name')
    output_file = config_parser.get('data', 'output_dir') + prodtest_name + ".pdf"
    with PdfPages(output_file) as pdf:
        first_page=plt.gcf()
        first_page.text(0.5, 0.5, prodtest_name, transform=first_page.transFigure,
                        size=24, ha="center")
        save_curr_figure_and_close(pdf)
        # Loop through the columns and test each
        columns = list(csv_file.columns.values)
        start_from_column = 2 #skip tslice and adserver_host
        for i in range(start_from_column, len(columns)) :
            column_name = columns[i]
            logger.info("Creating AdServer activity plot for {}".format(column_name))

            create_instance_behaviour_plot(pdf, column_name, CONTROL, control_group)
            create_instance_behaviour_plot(pdf, column_name, TEST, test_group)

            control_data = control_group[column_name]
            test_data = test_group[column_name]

            logger.info("Checking distribution type for {}".format(column_name))
            is_norm = is_normal_distribution(control_data) and is_normal_distribution(test_data)
            report_text = do_test(control_data, test_data, is_norm)

            logger.info("Creating distribution plot for {}".format(column_name))
            create_distribution_plot(pdf, column_name, control_data, test_data)
            add_report_text(pdf, column_name, report_text)
            logger.info("Creating box plot for {}".format(column_name))
            create_boxplot(pdf, column_name, control_data, test_data)
            logger.info('\n ================== \n')

    logger.info("Finished.")

# Generate the A/B test report in HTML format
def generate_html_report(control_group, test_group):
    logger.info("html report")

# Generate a report containing the plots and additional info about the A/B test
def generate_report(csv_file, control_group, test_group, output_type):
    if (output_type == 'pdf'):
        generate_pdf_report(csv_file, control_group, test_group)
    elif (output_type == 'html'):
        generate_html_report(control_group, test_group)
    else:
        raise Exception('Unsupported output file type for test report: ' + output_type)


if __name__ == "__main__":
    main()
