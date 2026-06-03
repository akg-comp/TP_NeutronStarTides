using Pkg
Pkg.add("PDFIO")
using PDFIO

function extract_text(filename, outfilename)
    doc = pdDocOpen(filename)
    open(outfilename, "w") do f
        for i in 1:pdDocGetPageCount(doc)
            page = pdDocGetPage(doc, i)
            pdPageExtractText(f, page)
        end
    end
    pdDocClose(doc)
end

extract_text("QNMLeaverExplained.pdf", "qnm_leaver.txt")
