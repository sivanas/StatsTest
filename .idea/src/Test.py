import pandas as pd
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import numpy as np
from numpy import linspace
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from enum import Enum


class OutputType(Enum):
    PDF = '/Users/sivan.as/Downloads/prodtestats.pdf'
    HTML = '/Users/sivan.as/Downloads/prodtestats.html'


OUTPUT_TYPE = OutputType.PDF
# OUTPUT_TYPE = OutputType.HTML

P_VALUE_PRECISION = 3
NORMAL_P_VALUE = 0.05

REPORT_TITLE = "Report for prodtest-ssp-12345-example"
CSV_INPUT_FILE_PATH = '/Users/sivan.as/Downloads/3ccf5ebf-d2c2-4f7b-ba0c-58b7feb2893d.csv'

TEST_SERVERS = [
    'i-0ce10dc8209c58a28',
    'i-0ddf0976483394643',
    'i-0828061826d38ad8d',
    'i-04ec2dc0f13cc8be8',
    'i-09f4bb20b2ccf8566',
    'i-042388cecaa737d56',
    'i-08be6e52c201af9f6',
    'i-0dd7d0ddcae6be9a4',
    'i-0b8357dea675a2cd1',
    'i-058d261d921da4b02'
]

COLUMN_NAMES = [
    'total_requests',
    'total_ex_bid_requests',
    'total_ex_valid_bid_requests',
    'total_ex_timeouts',
    'total_ex_nobids',
    'total_ex_errors',
    'total_ex_bids',
    'total_ex_valid_bids',
    'total_ex_discardedbids',
    'total_served',
    'fill_rate',
    'total_impressions',
    'total_ex_bid_impressions',
    'total_ex_bid_impression_errors',
    'ex_total_impression_price',
    'render_rate',
    'total_revenue',
    'ecpm',
    'total_clicks',
    'click_rate',
    'avg_request_duration',
    'avg_served_duration',
    'avg_click_duration'
]


# Main
def main():
    plot_config()
    csv_file = pd.read_csv(CSV_INPUT_FILE_PATH, skip_blank_lines=True)
    control_group = csv_file.loc[csv_file['adserver_host'].isin(TEST_SERVERS) == False]
    test_group = csv_file.loc[csv_file['adserver_host'].isin(TEST_SERVERS)]
    generate_report(control_group, test_group, OUTPUT_TYPE)


# Create design configuration for seaborn plot
def plot_config():
    sns.set(color_codes=True)
    sns.set_style('whitegrid')
    sns.set_palette('Set2')


# Check if the data is distributed normally
def is_normal_distribution(data):
    norm_res, norm_pval = stats.shapiro(data)
    is_normal_distribution = norm_pval > NORMAL_P_VALUE;
    return is_normal_distribution


# Run a stats test according to the distribution of the data
def do_test(control_data, test_data, is_normal_distribution):
    report_text = "Distribution:"
    if (is_normal_distribution):
        report_text += "The data is normally distributed \nRunning Test: T-Test\n"
        test_res, test_pval = stats.ttest_ind(control_data, test_data, axis=0, equal_var=False)
    else:
        report_text += "The data is NOT normally distributed \nRunning Test: Mann-Whitney U Test\n"
        test_res, test_pval = stats.mannwhitneyu(control_data, test_data)

    report_text += "Test Results: "

    is_significant_diff = test_pval < NORMAL_P_VALUE
    if (is_significant_diff):
        report_text += "The difference is significant with p-value {}".format(test_pval)
    else:
        report_text += "All Good!"

    return report_text


# Create a distribution plot from data
def create_distribution_plot(pdf, column_name, control_data, test_data):
    sns.distplot(control_data, label="Control")
    sns.distplot(test_data, label="Test")
    plt.legend()  # Show lables
    plt.title("Distribution for " + column_name)
    save_curr_figure_and_close(pdf)


def create_time_plot(pdf, column_name, data):
    adserver_data = pd.DataFrame(data=data).groupby(['tslice', 'adserver_host']).agg(column_name)
    adserver_data.sum().unstack().plot()
    plt.title(column_name + " Per AdServer")
    plt.xlabel("Time")
    plt.ylabel(column_name)
    save_curr_figure_and_close(pdf)

# Create a box plot from data
def create_boxplot(pdf, column_name, control_data, test_data):
    boxplot_data = pd.DataFrame({
        "Control": control_data,
        "Test": test_data
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
    plt.gcf().text(0.05, 0.8, "Evaluating {} \n".format(column_name), transform=plt.gcf().transFigure, size=18)
    plt.gcf().text(0.05, 0.7, text, transform=plt.gcf().transFigure)
    save_curr_figure_and_close(pdf)


# Generate the A/B test report in PDF format
def generate_pdf_report(control_group, test_group):
    with PdfPages(OutputType.PDF.value) as pdf:
        # Add Document Title
        plt.gcf().text(0.5, 0.5, REPORT_TITLE, transform=plt.gcf().transFigure,
                       size=24, ha="center")
        save_curr_figure_and_close(pdf)

        # Loop through the columns and test each
        for column_name in COLUMN_NAMES:

            create_time_plot(pdf, column_name, control_group)
            create_time_plot(pdf, column_name, test_group)

            control_data = control_group[column_name]
            test_data = test_group[column_name]

            # print "Control: \n {} \n Test: \n {} \n ".format(control_data, test_data)

            is_norm = is_normal_distribution(control_data) and is_normal_distribution(test_data)
            report_text = do_test(control_data, test_data, is_norm)

            add_report_text(pdf, column_name, report_text)
            create_distribution_plot(pdf, column_name, control_data, test_data)
            create_boxplot(pdf, column_name, control_data, test_data)


# Generate the A/B test report in HTML format
def generate_html_report(control_group, test_group):
    print "html report"


# Generate a report containing the plots and additional info about the A/B test
def generate_report(control_group, test_group, output_type):
    if (output_type == OutputType.PDF):
        generate_pdf_report(control_group, test_group)
    elif (output_type == OutputType.HTML):
        generate_html_report(control_group, test_group)
    else:
        raise Exception('Unsupported output file type for test report: ' + output_type)


if __name__ == "__main__":
    main()
