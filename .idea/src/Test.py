import ConfigParser
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt, mpld3
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

    generate_report(csv_file, control_group, test_group)
    logger.info("Finished.")

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
    is_ok = not is_significant_diff
    return is_ok, report_text


# Create a plot with the behaviour or each instance
def create_instance_behaviour_plot(column_name, group_name, data):
    fig, ax = plt.subplots()
    adserver_data = pd.DataFrame(data=data).groupby(['tslice', 'adserver_host'])
    adserver_data.agg(column_name).sum().unstack().plot(ax=ax)
    plt.title("{} Instances Activity for {}".format(group_name, column_name))
    plt.xlabel("Time")
    plt.ylabel(column_name)
    plt.close()
    return fig


# Create a distribution plot from data
def create_distribution_plot(column_name, control_data, test_data):
    fig, ax = plt.subplots()
    sns.distplot(control_data, label=CONTROL)
    sns.distplot(test_data, label=TEST)
    plt.legend()  # Show lables
    plt.title("Distribution for {}".format(column_name))
    plt.close()
    return fig


# Create a box plot from data
def create_boxplot(column_name, control_data, test_data):
    boxplot_data = pd.DataFrame({
        CONTROL : control_data,
        TEST : test_data
    })
    fig, ax = plt.subplots()
    sns.boxplot(data=boxplot_data, orient="v")
    plt.xlabel("Experiment Groups")
    plt.ylabel(column_name)
    plt.close()
    return fig

# Create figure with report text to current pdf figure
def create_report_text(column_name, text):
    fig, ax = plt.subplots()
    plt.grid(False)
    plt.axis('off')
    fig.text(0.5, 0.85, "Evaluating {}".format(column_name), size=18, ha="center")
    fig.text(0.05, 0.1, text, size=10)
    plt.close()
    return fig


# Create figure with report summary
def create_report_summary(prodtest_name, summary) :
    fig, ax = plt.subplots()
    #plt.subplots_adjust(left=0.2)
    plt.grid(False)
    plt.axis('off')
    fig.text(0.5, 0.85, prodtest_name+"\n", size=24, ha="center")
    fig.text(0.05, 0.8, "Summary:", size=14)
    summary_text = "\n"
    for column in summary:
        summary_text += column + " : " + summary[column] + "\n"
    fig.text(0.05, 0.1, summary_text, size=10)
    plt.close()
    return fig

# Generate report data - plots and text
def generate_report_data(csv_file, prodtest_name, control_group, test_group):
    plots = list()
    summary = {}
    enable_instance_report = config_parser.getboolean('data', 'enable_instance_report') == True
    columns = list(csv_file.columns.values)
    start_from_column = 2 #skip tslice and adserver_host

    # Loop through the columns and create plots and test for each
    for i in range(start_from_column, len(columns)) :
        column_name = columns[i]

        control_data = control_group[column_name]
        test_data = test_group[column_name]

        logger.info("Checking distribution type for {}".format(column_name))
        is_norm = is_normal_distribution(control_data) and is_normal_distribution(test_data)
        is_ok, test_info = do_test(control_data, test_data, is_norm)
        summary[column_name] = "OK!" if is_ok else "NOT OK!"
        plots.append(create_report_text(column_name, test_info))

        logger.info("Creating box plot for {}".format(column_name))
        plots.append(create_boxplot(column_name, control_data, test_data))

        logger.info("Creating distribution plot for {}".format(column_name))
        plots.append(create_distribution_plot(column_name, control_data, test_data))

        if enable_instance_report:
            logger.info("Creating AdServer activity plot for {}".format(column_name))
            plots.append(create_instance_behaviour_plot(column_name, CONTROL, control_group))
            plots.append(create_instance_behaviour_plot(column_name, TEST, test_group))

        logger.info('\n ================== \n')

    summary_fig = create_report_summary(prodtest_name, summary)
    return summary_fig, plots


# Generate a report containing the plots and additional info about the A/B test
def generate_report(csv_file, control_group, test_group):
    output_type = config_parser.get('data', 'output_type')
    prodtest_name = config_parser.get('data', 'prodtest_name')
    output_dir = config_parser.get('data', 'output_dir')

    summary_fig, plots = generate_report_data(csv_file, prodtest_name, control_group, test_group)

    if (output_type == 'pdf'):
        logger.info("Generating PDF Report...")
        output_file = output_dir + prodtest_name + ".pdf"
        with PdfPages(output_file) as pdf:
             #Add summary first
            pdf.savefig(summary_fig)
            # Add all figures
            for p in plots :
                pdf.savefig(p)

    elif (output_type == 'html'):
        logger.info("Generating HTML Report...")
        tmp_html_dir = output_dir + prodtest_name + "_html/"
        if not os.path.exists(tmp_html_dir):
            os.makedirs(tmp_html_dir)

        output_file = output_dir + prodtest_name + ".html"
        html = '<html>'
        #Add summary first
        summary_fig.savefig(tmp_html_dir + "summary.png")
        html += "<img src=\"{}\"><br>".format(tmp_html_dir + "summary.png")
        i = 1
        for p in plots :
            fig_path = "{}{}.{}".format(tmp_html_dir, i, "png")
            p.savefig(fig_path)
            html += "<img src=\"{}\"><br>".format(fig_path)
            i = i+1

        html += '</html>'
        html_file= open(output_file,"w")
        html_file.write(html)
        html_file.close()

    else:
        raise Exception('Unsupported output file type for test report: ' + output_type)



if __name__ == "__main__":
    main()
