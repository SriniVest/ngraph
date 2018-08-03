// This script acts as a trigger script for the main ngraph-unittest.groovy
// Jenkins job.  This script is part of a Jenkins multi-branch pipeline job
// which can trigger GitHub jobs more effectively than the GitHub Pull
// Request Builder (GHPRB) plugin, in our environment.

// The original ngraph-unittest job required the following parameters.  We
// set these up below as global variables, so we do not need to rewrite the
// original script -- we only need to provide this new trigger hook.
//
// ngraph-unittest parameters:
BRANCH = BRANCH_NAME
PR_URL = CHANGE_URL
PR_COMMIT_AUTHOR = CHANGE_AUTHOR
// TRIGGER_URL        <- No longer needed
JENKINS_BRANCH = "chrisl/new-ci-trigger"
TIMEOUTTIME = 3600

// Constants
JENKINS_DIR="."

// This groovy script is specifically designed as a Jenkins multi-branch
// pipeline trigger.  This requires some of the git checkout commands to
// be different ("checkout scm") than regular pipeline scripts (which use
// checkout(...) with explicit parameters).  Since some of the ngraph-unittest
// are dynamic functions shared by multiple jobs, we need a way to switch
// between the two types of checkouts.  Nick and I decided to use an optional
// environment variable -- if this environment variable exists when
// runNgraphBuild() runs, then "checkout scm" is used in order to be compatible
// with multi-branch pipeline checkouts.
env.MB_PIPELINE_CHECKOUT = true

node("bdw && nogpu") {

    deleteDir()  // Clear the workspace before starting

    echo "jenkins-trigger parameters:"
    echo "BRANCH           = ${BRANCH}"
    echo "PR_URL           = ${PR_URL}"
    echo "PR_COMMIT_AUTHOR = ${PR_COMMIT_AUTHOR}"
    echo "JENKINS_BRANCH   = ${JENKINS_BRANCH}"
    echo "TIMEOUTTIME      = ${TIMEOUTTIME}"

    // Clone the cje-algo directory which contains our Jenkins groovy scripts
    git(branch: JENKINS_BRANCH, changelog: false, poll: false,
        url: 'https://github.intel.com/AIPG/cje-algo')
    echo "After cloning cje-algo, workspace looks like:"
    sh "ls -l"

    // Call the main job script.
    //
    // NOTE: We keep the main job script in github.intel.com because it may
    //      contain references to technology which has not yet been released.
    //
    echo "Calling ngraph-unittest.groovy"
    returnValue = load("${JENKINS_DIR}/ngraph-unittest.groovy")
    echo "ngraph-unittest.groovy returned ${returnValue}"
    // def ngraphUnitTest = load("${JENKINS_DIR}/ngraph-unittest.groovy")
    // ngraphUnitTest()

}  // End:  node( ... )

echo "Done"

